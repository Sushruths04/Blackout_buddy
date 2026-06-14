# Modal-only fine-tuning

Training is intentionally separated from the offline inference application. Do not
run training on the Hugging Face Space or a local laptop.

## 1. Configure Modal

```powershell
python -m pip install modal==1.5.0
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_your_write_token
```

The token needs write access to the target Hugging Face model repository.

## 2. Train and publish

```powershell
modal run finetune/modal_train.py --output-repo your-name/blackout-buddy-nemotron-lora
```

The job uses an `A100-40GB` with `L40S` fallback, persists checkpoints in the
`blackout-buddy-training` Modal Volume, and uploads the final LoRA adapter to the
Hub. It packages `data/emergency_kb.json` and creates seven deterministic prompt
variants per card inside the remote container. The function has a three-hour
timeout and defaults to three epochs.

The same Modal image pins llama.cpp release `b9627` and converts the PEFT adapter
to `blackout-buddy-lora-f16.gguf` for the local llama.cpp runtime. Both formats
are uploaded to the target repository. If conversion fails, the PEFT adapter is
still published with `conversion_error.txt` for diagnosis.

## Important

- The base training model is `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.
- Training and adapter conversion run on Modal; no training runs locally.
- The production runtime remains llama.cpp and never calls Modal.
- Publishing an adapter does not make medical guidance validated. Keep retrieval
  and the verified-card fallback enabled.
