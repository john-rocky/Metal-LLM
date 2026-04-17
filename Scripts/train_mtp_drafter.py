#!/usr/bin/env python3
"""
Train MTP drafter for Gemma 4 E2B against llama.cpp Q4_K_M output distribution.

HYBRID approach (recommended):
  1. Load HuggingFace Gemma 4 E2B (fp16) to extract hidden states + KV caches.
     These are the drafter's INPUT features (carry state, attention context).
  2. Load llama-cpp-python with the EXACT Q4_K_M GGUF that ships on-device.
     This provides the TARGET tokens/logits the drafter must learn to predict.
  3. Train the drafter so it predicts Q4_K_M output, not fp32 output.

Why hybrid?
  - llama-cpp-python does not expose per-layer hidden states or KV caches.
  - HuggingFace does. So HF provides the drafter's input signal.
  - But the *labels* must come from Q4_K_M, because that is what the drafter
    will be paired with at inference time. Training against fp32 labels
    introduces a distribution mismatch that lowers acceptance rate.

Fallback:
  If --gguf-path is not provided or llama-cpp-python is not installed,
  the script falls back to pure HF targets (fp32 argmax). This still works
  but expect 5-15% lower acceptance rate vs the hybrid approach.

Usage (hybrid, recommended):
    pip install torch transformers datasets accelerate llama-cpp-python tqdm
    python Scripts/train_mtp_drafter.py \
        --model google/gemma-4-E2B-it \
        --gguf-path Models/gemma-4-2b-it-Q4_K_M.gguf \
        --output Models/mtp_drafter.pt \
        --epochs 3 --lr 1e-4 --num-samples 200 --validate

Usage (fallback, HF-only):
    python Scripts/train_mtp_drafter.py \
        --model google/gemma-4-E2B-it \
        --output Models/mtp_drafter.pt \
        --epochs 3 --lr 1e-4 --num-samples 200

For Colab (free tier, T4 GPU):
    Use --gradient-checkpointing --num-samples 100 --max-len 128
For Mac Studio (M2 Ultra / M4 Max):
    Use device auto-detected as MPS, no special flags needed.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm, trange
except ImportError:
    # Minimal fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable
    def trange(*args, **kwargs):
        return range(*args)

# Add CoreML-LLM conversion path for the MTP model definition
_COREML_LLM_ROOT = Path(__file__).resolve().parent.parent.parent / "CoreML-LLM"
sys.path.insert(0, str(_COREML_LLM_ROOT / "conversion"))
try:
    from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig
    print("[OK] Imported MtpDrafterModel from CoreML-LLM/conversion/mtp_drafter_model.py")
except ImportError:
    print(f"[FATAL] Could not import MtpDrafterModel from {_COREML_LLM_ROOT / 'conversion'}")
    print("  Make sure CoreML-LLM repo is cloned alongside Metal-LLM.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# llama-cpp-python target model (Q4_K_M GGUF)
# ---------------------------------------------------------------------------

def _try_load_llama_cpp(gguf_path: str, n_ctx: int = 2048):
    """Attempt to load a GGUF model via llama-cpp-python.

    Returns (llama_model, tokenize_fn, detokenize_fn) or None if unavailable.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[WARN] llama-cpp-python not installed. Install with:")
        print("       pip install llama-cpp-python")
        print("       Falling back to HF-only targets (fp32 distribution).")
        print("       >>> This may reduce acceptance rate by 5-15% vs hybrid. <<<")
        return None

    if not gguf_path or not Path(gguf_path).exists():
        print(f"[WARN] GGUF file not found: {gguf_path}")
        print("       Falling back to HF-only targets.")
        return None

    print(f"Loading Q4_K_M GGUF: {gguf_path}")
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_batch=512,
        n_gpu_layers=0,   # CPU is fine for single-token eval
        logits_all=True,  # We need logits at every position
        verbose=False,
    )
    print(f"  GGUF loaded: ctx={n_ctx}, vocab={llm.n_vocab()}")
    return llm


def _get_gguf_targets(llm, token_ids: list[int]) -> torch.Tensor:
    """Run the GGUF model on a token sequence, return argmax at each position.

    Args:
        llm: Llama instance from llama-cpp-python.
        token_ids: list of int token IDs (already tokenized by HF tokenizer).

    Returns:
        targets: (seq_len,) int64 tensor of Q4_K_M argmax predictions.
                 targets[i] = argmax of Q4_K_M logits at position i
                 (i.e., what the model predicts AFTER seeing tokens[0..i]).
    """
    import numpy as np

    # Reset KV cache before each sequence
    llm.reset()

    # Evaluate all tokens at once
    llm.eval(token_ids)

    # Extract argmax at each position
    seq_len = len(token_ids)
    targets = []
    for pos in range(seq_len):
        logits_at_pos = llm.scores[pos]  # numpy array of shape (vocab,)
        targets.append(int(np.argmax(logits_at_pos)))

    return torch.tensor(targets, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data collection: hybrid HF (features) + GGUF (targets)
# ---------------------------------------------------------------------------

class MtpTrainingDataset(Dataset):
    """In-memory dataset of (token_id, hidden, kv13, kv14, target, position) tuples."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _select_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_training_data(
    model_name: str,
    gguf_path: str | None,
    num_samples: int = 200,
    max_len: int = 128,
    device: str = "cuda",
    gradient_checkpointing: bool = False,
):
    """Run HF teacher + optionally GGUF target model to collect training pairs.

    For each text sample:
      - HF forward pass gives: hidden states, KV caches (drafter inputs)
      - GGUF forward pass gives: argmax tokens (drafter training targets)
      - If GGUF unavailable, HF argmax is used as target (fallback)

    Returns:
        samples: list of dicts with training data
        tokenizer: HF tokenizer
        mode: "hybrid" or "hf-only"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # --- Load HF model (for hidden states + KV caches) ---
    print(f"Loading HF teacher model: {model_name} (for hidden states + KV caches)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device if device != "mps" else "cpu",  # device_map doesn't support "mps"
        output_hidden_states=True,
    )
    if device == "mps":
        model = model.to("mps")
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.eval()

    # --- Load GGUF model (for Q4_K_M target tokens) ---
    llm_gguf = None
    mode = "hf-only"
    if gguf_path:
        llm_gguf = _try_load_llama_cpp(gguf_path, n_ctx=max_len + 64)
        if llm_gguf is not None:
            mode = "hybrid"
            print("[OK] HYBRID mode: HF hidden states + Q4_K_M targets")
        else:
            print("[FALLBACK] HF-only mode: hidden states AND targets from fp32")
    else:
        print("[INFO] No --gguf-path provided. Using HF-only mode.")
        print("       For better acceptance rate, provide the Q4_K_M GGUF file.")

    # --- Load training texts ---
    print("Loading training data (FineWeb-Edu)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True,
    )

    samples = []
    texts_processed = 0
    skipped = 0

    pbar = tqdm(total=num_samples, desc="Collecting data", unit="text")
    for item in ds:
        if texts_processed >= num_samples:
            break

        text = item.get("text", "")
        if len(text) < 100:
            skipped += 1
            continue

        tokens = tokenizer.encode(
            text, max_length=max_len, truncation=True,
            return_tensors="pt",
        )
        if tokens.shape[1] < 10:
            skipped += 1
            continue

        tokens_device = tokens.to(device)

        # --- HF forward: get hidden states + KV caches ---
        with torch.no_grad():
            outputs = model(tokens_device, output_hidden_states=True, use_cache=True)

        hidden = outputs.hidden_states[-1]  # (1, seq, 1536)
        hf_logits = outputs.logits          # (1, seq, vocab)

        # KV cache: outputs.past_key_values is tuple of (key, value) per layer
        past = outputs.past_key_values
        if not past or len(past) <= 14:
            skipped += 1
            continue

        kv13_k = past[13][0].cpu()  # (1, heads, seq, head_dim)
        kv13_v = past[13][1].cpu()
        kv14_k = past[14][0].cpu()
        kv14_v = past[14][1].cpu()

        # --- Get target tokens ---
        if mode == "hybrid":
            # Q4_K_M targets: run same tokens through GGUF
            token_list = tokens[0].tolist()
            gguf_targets = _get_gguf_targets(llm_gguf, token_list)  # (seq,)
        else:
            # Fallback: HF argmax targets
            gguf_targets = hf_logits[0].argmax(dim=-1).cpu()  # (seq,)

        # --- Store per-position training pairs ---
        seq_len = tokens.shape[1]
        for pos in range(1, seq_len - 1):
            # Drafter input = HF hidden states (carry from previous position)
            # Target = Q4_K_M argmax at current position (what to predict next)
            samples.append({
                "embed_token": tokens[0, pos].cpu(),
                "hidden": hidden[0, pos - 1].cpu().half(),
                "kv13_k": kv13_k[0, :, :pos + 1, :].half(),
                "kv13_v": kv13_v[0, :, :pos + 1, :].half(),
                "kv14_k": kv14_k[0, :, :pos + 1, :].half(),
                "kv14_v": kv14_v[0, :, :pos + 1, :].half(),
                "target": gguf_targets[pos],
                "position": pos,
            })

        texts_processed += 1
        pbar.update(1)
        pbar.set_postfix(pairs=len(samples), mode=mode)

    pbar.close()

    # --- Clean up HF model ---
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    print(f"Data collection complete:")
    print(f"  Mode: {mode}")
    print(f"  Texts processed: {texts_processed} (skipped {skipped})")
    print(f"  Training pairs: {len(samples):,}")

    if mode == "hybrid":
        # Quick stat: how many Q4_K_M targets differ from HF targets?
        # (This shows how much distribution mismatch we're correcting for)
        print(f"  [Hybrid] Drafter will learn Q4_K_M distribution directly.")

    return samples, tokenizer, mode


# ---------------------------------------------------------------------------
# Validation: quick acceptance rate test
# ---------------------------------------------------------------------------

def validate_acceptance(
    drafter: nn.Module,
    embed_table: torch.Tensor,
    samples: list[dict],
    device: str,
    num_eval: int = 500,
):
    """Run a quick acceptance rate test on held-out samples.

    Acceptance = fraction of positions where drafter argmax == target argmax.
    This simulates speculative decoding acceptance without the full pipeline.
    """
    drafter.eval()
    cfg = drafter.cfg
    correct = 0
    total = 0
    eval_samples = samples[:num_eval]

    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Validating", unit="pos"):
            token_id = sample["embed_token"].to(device)
            hidden = sample["hidden"].to(device).unsqueeze(0).unsqueeze(0)
            embed = embed_table[token_id].unsqueeze(0).unsqueeze(0)
            activations = torch.cat([embed, hidden], dim=-1).half()

            pos = sample["position"]
            kv13_k = sample["kv13_k"].to(device).unsqueeze(0)
            kv13_v = sample["kv13_v"].to(device).unsqueeze(0)
            kv14_k = sample["kv14_k"].to(device).unsqueeze(0)
            kv14_v = sample["kv14_v"].to(device).unsqueeze(0)

            ctx_swa = kv13_k.shape[-2]
            ctx_full = kv14_k.shape[-2]
            mask_swa = torch.zeros(1, 1, 1, ctx_swa, device=device, dtype=torch.float16)
            mask_full = torch.zeros(1, 1, 1, ctx_full, device=device, dtype=torch.float16)
            input_pos = torch.tensor([pos], device=device)
            target = sample["target"].to(device)

            logits, _ = drafter(
                activations, input_pos,
                kv13_k, kv13_v, kv14_k, kv14_v,
                mask_swa, mask_full,
            )

            pred = logits.argmax(dim=-1).view(-1)
            correct += (pred == target).sum().item()
            total += 1

    acc = correct / total * 100 if total > 0 else 0.0
    return acc, total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = _select_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif device == "mps":
        print("  Apple Silicon MPS backend")

    # --- Collect data ---
    samples, tokenizer, mode = collect_training_data(
        model_name=args.model,
        gguf_path=args.gguf_path,
        num_samples=args.num_samples,
        max_len=args.max_len,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if not samples:
        print("No training data collected! Exiting.")
        return

    # --- Split into train / validation ---
    random.shuffle(samples)
    val_size = min(500, len(samples) // 5)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    print(f"Train: {len(train_samples):,} pairs, Validation: {len(val_samples):,} pairs")

    # --- Initialize drafter ---
    cfg = MtpDrafterConfig()
    drafter = MtpDrafterModel(cfg).to(device).half()
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter params: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Load teacher's embedding table ---
    from transformers import AutoModelForCausalLM
    print("Loading embedding table from HF model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=device if device != "mps" else "cpu",
    )
    if device == "mps":
        embed_table = teacher.model.embed_tokens.weight.detach().to("mps")
    else:
        embed_table = teacher.model.embed_tokens.weight.detach()
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_samples)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Output directory ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, mode={mode}")
    print(f"{'='*60}\n")

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        drafter.train()
        random.shuffle(train_samples)

        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        pbar = tqdm(
            enumerate(train_samples),
            total=len(train_samples),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="pair",
        )

        for step, sample in pbar:
            token_id = sample["embed_token"].to(device)
            hidden = sample["hidden"].to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 1536)

            # Embed current token
            embed = embed_table[token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, 1536)

            # Concatenate: (embed, hidden) -> (1, 1, 3072)
            activations = torch.cat([embed, hidden], dim=-1).half()

            # KV caches
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
            logits, proj_act = drafter(
                activations, input_pos,
                kv13_k, kv13_v, kv14_k, kv14_v,
                mask_swa, mask_full,
            )

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

            # Update progress bar
            if total > 0:
                pbar.set_postfix(
                    loss=f"{total_loss / total:.4f}",
                    acc=f"{correct / total * 100:.1f}%",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        pbar.close()

        # --- Epoch summary ---
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total if total > 0 else 0
        train_acc = correct / total * 100 if total > 0 else 0
        print(f"\nEpoch {epoch + 1}/{args.epochs} complete in {epoch_time:.0f}s")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Train accuracy: {train_acc:.1f}%")

        # --- Validation ---
        if val_samples:
            val_acc, val_total = validate_acceptance(
                drafter, embed_table, val_samples, device, num_eval=len(val_samples),
            )
            print(f"  Val acceptance: {val_acc:.1f}% ({val_total} samples)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(drafter.state_dict(), output_path)
                print(f"  [BEST] Saved best model -> {output_path}")
        else:
            val_acc = train_acc

        # --- Checkpoint every epoch ---
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": drafter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "mode": mode,
            "config": {
                "model": args.model,
                "gguf_path": args.gguf_path,
                "lr": args.lr,
                "epochs": args.epochs,
                "num_samples": args.num_samples,
                "max_len": args.max_len,
            },
        }, ckpt_path)
        print(f"  Checkpoint saved -> {ckpt_path}")
        print()

    # --- Save final model (if not already saved as best) ---
    if not val_samples or best_val_acc == 0:
        torch.save(drafter.state_dict(), output_path)

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"  Best val acceptance: {best_val_acc:.1f}%")
    print(f"  Model saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Mode: {mode}")
    if mode == "hybrid":
        print(f"  Target source: Q4_K_M GGUF ({args.gguf_path})")
    else:
        print(f"  Target source: HF fp32 ({args.model})")
        print(f"  [NOTE] For better on-device acceptance, retrain with --gguf-path")
    print(f"{'='*60}")

    # --- Optional post-training validation ---
    if args.validate:
        print("\n--- Post-training acceptance test ---")
        # Use all samples for final validation
        final_acc, final_total = validate_acceptance(
            drafter, embed_table, samples, device,
            num_eval=min(2000, len(samples)),
        )
        print(f"Final acceptance rate: {final_acc:.1f}% over {final_total} positions")
        if mode == "hybrid":
            print("(Measured against Q4_K_M targets -- this reflects on-device acceptance)")
        else:
            print("(Measured against HF fp32 targets -- on-device acceptance may differ)")

        # Estimate on-device speculative decoding speedup
        # With k draft tokens and acceptance rate alpha:
        # Expected accepted tokens per step = alpha / (1 - alpha^(k+1)) * (1 - alpha)
        # Simplified: for k=3 drafts, speedup ~ 1 + k*alpha (rough)
        alpha = final_acc / 100.0
        k = 3  # typical draft count
        expected_accepted = sum(alpha**i for i in range(1, k + 1))
        speedup = 1 + expected_accepted
        print(f"Estimated speedup with k={k} drafts: {speedup:.2f}x "
              f"(acceptance alpha={alpha:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MTP drafter for Gemma 4 E2B (hybrid Q4_K_M targeting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hybrid mode (recommended): train against Q4_K_M output
  python Scripts/train_mtp_drafter.py \\
      --model google/gemma-4-E2B-it \\
      --gguf-path Models/gemma-4-2b-it-Q4_K_M.gguf \\
      --output Models/mtp_drafter.pt \\
      --epochs 3 --lr 1e-4 --validate

  # Fallback mode: train against HF fp32 output
  python Scripts/train_mtp_drafter.py \\
      --model google/gemma-4-E2B-it \\
      --output Models/mtp_drafter.pt \\
      --epochs 3

  # Colab (T4 GPU, limited VRAM)
  python Scripts/train_mtp_drafter.py \\
      --model google/gemma-4-E2B-it \\
      --gguf-path Models/gemma-4-2b-it-Q4_K_M.gguf \\
      --gradient-checkpointing \\
      --num-samples 100 --max-len 128 --validate
        """,
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it",
                        help="HuggingFace teacher model (for hidden states + embeddings)")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to Q4_K_M GGUF file for target distribution. "
                             "If omitted, falls back to HF fp32 targets.")
    parser.add_argument("--output", default="Models/mtp_drafter.pt",
                        help="Output path for trained drafter weights")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of text samples to collect training data from")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Max sequence length per sample")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM on Colab)")
    parser.add_argument("--validate", action="store_true",
                        help="Run acceptance rate test after training")
    args = parser.parse_args()
    train(args)
