#!/usr/bin/env python3
"""
Train MTP drafter for Gemma 4 E2B against llama.cpp Q4_K_M output distribution.

Strategy:
  1. Load Gemma 4 E2B from HuggingFace (fp32 teacher — we use this to get hidden states)
  2. For each training sample, run teacher forward to get:
     - Hidden states at each position (for carry state)
     - KV caches (kv13, kv14) from attention layers
     - Next-token logits (training targets)
  3. Train 4-layer MTP drafter to predict teacher's next-token given:
     - embed(current_token) + projected_activations
     - Teacher's KV caches
  4. Export trained drafter as .pt, then convert to CoreML/GGUF

Key insight: We train against HuggingFace fp32 Gemma 4 (same as llama.cpp's source model).
The Q4_K_M quantization introduces small perturbations, but the distributions are close enough
that a drafter trained on fp32 should achieve >50% acceptance on Q4_K_M.

Unlike Google's approach (trained against W4A8), we train against the CANONICAL fp32 forward.
llama.cpp's Q4_K_M dequant is designed to closely approximate fp32, so this should transfer.

Usage:
    pip install torch transformers datasets accelerate
    python Scripts/train_mtp_drafter.py \
        --model google/gemma-4-E2B-it \
        --output Models/mtp_drafter.pt \
        --epochs 3 --batch-size 4 --lr 1e-4

For Colab (free tier, T4 GPU):
    Use --gradient-checkpointing --batch-size 1 --gradient-accumulation 8
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add CoreML-LLM conversion path for the MTP model definition
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CoreML-LLM" / "conversion"))
try:
    from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig
    print("[OK] Imported MtpDrafterModel from CoreML-LLM")
except ImportError:
    print("[WARN] Could not import from CoreML-LLM, using inline definition")
    # Inline minimal definition (same architecture)
    MtpDrafterModel = None


# ---------------------------------------------------------------------------
# Data collection: run teacher model, collect hidden states + KV + targets
# ---------------------------------------------------------------------------

def collect_training_data(
    model_name: str,
    num_samples: int = 1000,
    max_len: int = 256,
    device: str = "cuda",
):
    """Run teacher and collect (hidden_state, kv13, kv14, next_token) tuples."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    # Load training texts
    print("Loading training data...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        text = item.get("text", "")
        if len(text) < 100:
            continue

        tokens = tokenizer.encode(text, max_length=max_len, truncation=True,
                                   return_tensors="pt").to(device)
        if tokens.shape[1] < 10:
            continue

        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True, use_cache=True)

        # Extract what the MTP drafter needs:
        # - Hidden states from last layer (for carry state / projected activations)
        # - KV cache from layer 13 (SWA) and layer 14 (full attention)
        # - Next-token targets (argmax of output logits)
        hidden = outputs.hidden_states[-1]  # (1, seq, 1536)
        logits = outputs.logits             # (1, seq, vocab)
        targets = logits.argmax(dim=-1)     # (1, seq)

        # KV cache: outputs.past_key_values is a tuple of (key, value) per layer
        past = outputs.past_key_values
        if past and len(past) > 14:
            kv13_k = past[13][0].cpu()  # (1, heads, seq, head_dim)
            kv13_v = past[13][1].cpu()
            kv14_k = past[14][0].cpu()
            kv14_v = past[14][1].cpu()
        else:
            continue

        # Store per-position training pairs
        for pos in range(1, tokens.shape[1] - 1):
            samples.append({
                "embed_token": tokens[0, pos].cpu(),           # current token id
                "hidden": hidden[0, pos - 1].cpu().half(),     # previous hidden (carry)
                "kv13_k": kv13_k[0, :, :pos+1, :].cpu().half(),
                "kv13_v": kv13_v[0, :, :pos+1, :].cpu().half(),
                "kv14_k": kv14_k[0, :, :pos+1, :].cpu().half(),
                "kv14_v": kv14_v[0, :, :pos+1, :].cpu().half(),
                "target": targets[0, pos].cpu(),               # next token
                "position": pos,
            })

        if (i + 1) % 10 == 0:
            print(f"  Collected {len(samples)} training pairs from {i+1} texts")

    del model
    torch.cuda.empty_cache()
    print(f"Total training pairs: {len(samples)}")
    return samples, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect data
    samples, tokenizer = collect_training_data(
        args.model, num_samples=args.num_samples,
        max_len=args.max_len, device=device
    )

    if not samples:
        print("No training data collected!")
        return

    # Initialize drafter
    cfg = MtpDrafterConfig()
    drafter = MtpDrafterModel(cfg).to(device).half()
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Load teacher's embedding table for token embedding lookup
    from transformers import AutoModelForCausalLM
    print("Loading embedding table...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    )
    embed_table = teacher.model.embed_tokens.weight.detach()  # (vocab, 1536)
    del teacher
    torch.cuda.empty_cache()

    # Optimizer
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(samples))

    # Training loop
    drafter.train()
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        import random
        random.shuffle(samples)

        for i, sample in enumerate(samples):
            token_id = sample["embed_token"].to(device)
            hidden = sample["hidden"].to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 1536)

            # Embed current token (unscaled, matching Google's approach)
            embed = embed_table[token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, 1536)

            # Concatenate: (embed, hidden) → (1, 1, 3072)
            activations = torch.cat([embed, hidden], dim=-1).half()

            # KV caches — pad to fixed context length
            pos = sample["position"]
            kv13_k = sample["kv13_k"].to(device).unsqueeze(0)
            kv13_v = sample["kv13_v"].to(device).unsqueeze(0)
            kv14_k = sample["kv14_k"].to(device).unsqueeze(0)
            kv14_v = sample["kv14_v"].to(device).unsqueeze(0)

            # Masks
            ctx_swa = kv13_k.shape[-2]
            ctx_full = kv14_k.shape[-2]
            mask_swa = torch.zeros(1, 1, 1, ctx_swa, device=device, dtype=torch.float16)
            mask_full = torch.zeros(1, 1, 1, ctx_full, device=device, dtype=torch.float16)

            input_pos = torch.tensor([pos], device=device)
            target = sample["target"].to(device)

            # Forward
            logits, proj_act = drafter(activations, input_pos,
                                        kv13_k, kv13_v, kv14_k, kv14_v,
                                        mask_swa, mask_full)

            # Loss: cross-entropy on next token prediction
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1).view(-1)
            correct += (pred == target).sum().item()
            total += 1

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                acc = correct / total * 100
                print(f"  Epoch {epoch+1}/{args.epochs} Step {i+1}/{len(samples)} "
                      f"loss={avg_loss:.4f} acc={acc:.1f}%")

        avg_loss = total_loss / len(samples)
        acc = correct / total * 100
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} accuracy={acc:.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(drafter.state_dict(), output_path)
    print(f"Saved drafter to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTP drafter for Gemma 4 E2B")
    parser.add_argument("--model", default="google/gemma-4-E2B-it",
                        help="HuggingFace teacher model")
    parser.add_argument("--output", default="Models/mtp_drafter.pt",
                        help="Output path for trained drafter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of text samples to collect training data from")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Max sequence length per sample")
    args = parser.parse_args()
    train(args)
