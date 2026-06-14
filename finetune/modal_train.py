from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "blackout-buddy-lora"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
LLAMA_CPP_RELEASE = "b9627"
DEFAULT_OUTPUT_REPO = "alexz-oai/blackout-buddy-nemotron-lora"
VOLUME_PATH = Path("/vol")
ROOT = Path(__file__).resolve().parents[1]

PROMPT_TEMPLATES = [
    "What should I do right now for {title}?",
    "We have no internet and may not reach help quickly. Give safe first steps for {title}.",
    "Give a calm, short action plan for {title}. Include what not to do.",
    "Emergency field guide: {title}. What are the immediate priorities?",
    "Someone nearby may have {title}. Give only reviewed immediate actions and warnings.",
    "Explain the first five safe actions for {title} in plain language.",
    "I am panicking about {title}. Tell me what to do first, then what to avoid.",
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("blackout-buddy-training", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "huggingface-hub==1.19.0",
        "peft==0.19.1",
        "safetensors==0.8.0",
        "sentencepiece==0.2.1",
        "torch==2.12.0",
        "transformers==5.12.0",
    )
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_CPP_RELEASE} "
        "https://github.com/ggml-org/llama.cpp /opt/llama.cpp"
    )
    .add_local_file(
        ROOT / "data" / "emergency_kb.json",
        remote_path="/workspace/emergency_kb.json",
        copy=True,
    )
)


@app.function(
    image=image,
    gpu=["A100-40GB", "L40S"],
    timeout=60 * 60 * 3,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    output_repo: str = DEFAULT_OUTPUT_REPO,
    epochs: int = 3,
    learning_rate: float = 2e-4,
) -> dict[str, object]:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    token = os.environ["HF_TOKEN"]
    output_dir = VOLUME_PATH / "blackout-buddy-lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = json.loads(
        Path("/workspace/emergency_kb.json").read_text(encoding="utf-8")
    )

    def format_response(card: dict) -> str:
        steps = "\n".join(
            f"{index}. {step}"
            for index, step in enumerate(card["steps"], start=1)
        )
        avoid = "\n".join(f"- {item}" for item in card.get("avoid", []))
        return (
            f"ACT NOW\n{steps}\n\nAVOID\n{avoid}\n\n"
            f"Source: {card['source_name']}\n"
            "Reach trained help whenever any route is available."
        )

    rows = [
        {
            "instruction": template.format(title=card["title"]),
            "response": format_response(card),
            "source_card": card["id"],
        }
        for card in cards
        for template in PROMPT_TEMPLATES
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(row: dict) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Give calm, short, source-grounded emergency guidance. "
                    "Never invent a treatment or dosage. Preserve warnings."
                ),
            },
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
        )

    wrapped = Dataset.from_list(rows).shuffle(seed=42)
    tokenized = wrapped.map(
        lambda batch: tokenizer(
            [
                render(
                    {
                        "instruction": instruction,
                        "response": response,
                    }
                )
                for instruction, response in zip(
                    batch["instruction"],
                    batch["response"],
                    strict=True,
                )
            ],
            truncation=True,
            max_length=1536,
            padding=False,
        ),
        batched=True,
        remove_columns=wrapped.column_names,
        desc="Tokenizing reviewed emergency cards",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    gguf_path = output_dir / "blackout-buddy-lora-f16.gguf"
    conversion_error = None
    try:
        subprocess.run(
            [
                "python",
                "/opt/llama.cpp/convert_lora_to_gguf.py",
                "--base-model-id",
                BASE_MODEL,
                "--trust-remote-code",
                "--outfile",
                str(gguf_path),
                "--outtype",
                "f16",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        conversion_error = (
            f"Command failed with exit code {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}\n"
        )
        (output_dir / "conversion_error.txt").write_text(
            conversion_error,
            encoding="utf-8",
        )

    metadata = {
        "base_model": BASE_MODEL,
        "dataset_rows": len(rows),
        "epochs": epochs,
        "llama_cpp_release": LLAMA_CPP_RELEASE,
        "gguf_adapter": gguf_path.name if gguf_path.exists() else None,
        "gguf_conversion_error": conversion_error is not None,
        "training_platform": "Modal",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    volume.commit()

    HfApi(token=token).create_repo(
        repo_id=output_repo,
        repo_type="model",
        exist_ok=True,
    )
    HfApi(token=token).upload_folder(
        repo_id=output_repo,
        repo_type="model",
        folder_path=str(output_dir),
    )
    return {
        "output_repo": output_repo,
        "dataset_rows": len(rows),
        "volume_path": str(output_dir),
        "gguf_adapter": str(gguf_path) if gguf_path.exists() else None,
        "conversion_error": conversion_error,
    }


@app.local_entrypoint()
def main(
    output_repo: str = DEFAULT_OUTPUT_REPO,
    epochs: int = 3,
    learning_rate: float = 2e-4,
) -> None:
    result = train.remote(
        output_repo=output_repo,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    print(json.dumps(result, indent=2))
