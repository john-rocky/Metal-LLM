# Models

Model files are not tracked in git. Download the GGUF you want and drop it here.

## Gemma 4 E2B (primary target)

The repo's headline benchmark uses Gemma 4 E2B in INT4 GGUF form. Suggested sources:

- Google's official Gemma release on Hugging Face (convert to GGUF with `llama.cpp/convert_hf_to_gguf.py`, then quantise with `llama-quantize … Q4_K_M`).
- Community pre-converted GGUF if one is available and trustworthy.

Expected filename convention in Metal-LLM:

```
Models/gemma-4-e2b-q4_k_m.gguf
```

## ANE bridge (Week 7+)

The ANE prefill path reuses the chunk models from the sibling repo. See the
CoreML-LLM README for conversion instructions — no changes are needed on that
side; Metal-LLM reads the `.mlpackage` files directly.

## Smaller models for CI / smoke tests

Any small GGUF (TinyLlama-1.1B, Qwen2.5-0.5B) works for the load-and-tokenize
smoke tests introduced in Week 1.
