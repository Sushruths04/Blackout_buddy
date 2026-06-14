# Model inventory

Verified on June 14, 2026.

| Role | Repository | File | Runtime | License note |
|---|---|---|---|---|
| Primary | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` | `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | llama.cpp | NVIDIA Nemotron Open Model License |
| Multilingual | `CohereLabs/tiny-aya-global-GGUF` | `tiny-aya-global-q4_k_m.gguf` | llama.cpp | CC BY-NC 4.0 |
| LoRA base | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | Hub safetensors | Modal training only | NVIDIA Nemotron Open Model License |

## Runtime policy

The app first checks the explicit model paths in `.env`, then the local Hugging
Face cache. It does not download a model unless
`BLACKOUT_ALLOW_MODEL_DOWNLOAD=1` is explicitly set.

If no compatible GGUF is present, Blackout Buddy starts in Safe Field-Guide mode.
All quick cards, retrieval, triage, sources, and the UI remain available.

## Fine-tuned adapter

The Modal job publishes both the standard PEFT adapter and
`blackout-buddy-lora-f16.gguf`, converted with pinned llama.cpp release `b9627`.
Download the GGUF adapter beside the primary model, then set:

```powershell
$env:BLACKOUT_LORA_PATH="D:\models\blackout-buddy-lora.gguf"
```

The adapter is attached only to the primary NVIDIA model. If the file is absent,
the base GGUF still loads. If conversion fails for a future model revision, the
published repository includes `conversion_error.txt`; the PEFT adapter remains
available for a corrected conversion run.

## License consequence

The Cohere model is non-commercial (`CC BY-NC 4.0`). Keep it optional and review
submission and downstream distribution terms before claiming commercial use.

## Sources

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
- https://huggingface.co/CohereLabs/tiny-aya-global-GGUF
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
