#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL logit-lens over image patches, per generated token.

For each generated token step:
  For each hidden-state index (embedding + all transformer layers):
    For each image token (patch) in the visual span:
      - Compute top-K vocab predictions (final-norm -> lm_head -> softmax).
      - Compute digit (0..9) probabilities from the full-vocab softmax
        (both absolute and renormalized over digits only).
      - Store explicit patch location as (row, col) and its linear index.

Writes one JSON per image with a compact, structured schema (no attention, no extra artifacts).
All comments are in English as requested.
"""

import os
import glob
import json
from typing import List, Dict, Any, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ========================= CONFIG =========================
MODEL_DIR   = "/home/mmd/qwen/models/Qwen2.5-VL-7B-Instruct"
IMAGES_DIR  = "/home/mmd/qwen/logit_lens/data/images"
OUTPUT_DIR  = "logitlens_json"    # where JSONs are written
QUESTION    = "How many circles are there in this image?"
MAX_NEW_TOKENS = 512

# Logit-lens parameters
TOPK = 5                   # top-k vocab entries to store per (layer_idx, patch)
INCLUDE_EMBED_IDX = True   # include hidden_states[0] (embedding) as layer_idx=0
# =========================================================


# -------------------- Vision helper --------------------
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    # Fallback: load from path(s) directly
    def process_vision_info(messages: List[dict]):
        imgs, sizes = [], []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    path = part["image"]
                    img = Image.open(path).convert("RGB")
                    imgs.append(img)
                    sizes.append((img.height, img.width))
        return imgs, sizes


# -------------------- IO helpers --------------------
def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")
    files: List[str] = []
    for p in exts:
        files.extend(glob.glob(os.path.join(folder, p)))
    return sorted(files)


# -------------------- Model / Processor --------------------
def load_model_and_processor(model_dir: str):
    """
    Load Qwen2.5-VL model + processor.
    Uses bf16 if CUDA is available, else fp32.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",   # harmless here; useful if you later want attentions
        local_files_only=os.path.isdir(model_dir),
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=os.path.isdir(model_dir)
    )
    model.eval()
    return model, processor, device


# -------------------- Prompt packer --------------------
def build_messages(image_path: str, question: str) -> List[dict]:
    """
    Pack a single-turn user message with one image and a text question.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 512*28*28},
            {"type": "text", "text": question},
        ],
    }]


# -------------------- Grid derivation (rows/cols of patches) --------------------
def derive_grid(processor, image_inputs, model) -> Tuple[int, int, int, int, int]:
    """
    Recover merged-patch grid geometry from the processor+model config.
    Returns:
      grid_h, grid_w           : number of merged patches along H/W
      merged_patch_px          : patch size in resized pixels (after merging)
      resized_w, resized_h     : canvas size fed into the vision encoder
    """
    meta = processor.image_processor(images=image_inputs)
    thw = meta["image_grid_thw"].to("cpu").numpy().squeeze(0)  # [T, H_raw, W_raw]
    patch_size = int(getattr(model.config.vision_config, "patch_size", 14))
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    grid_h = int(thw[1] // merge_size)
    grid_w = int(thw[2] // merge_size)
    resized_h = int(thw[1] * patch_size)
    resized_w = int(thw[2] * patch_size)
    merged_patch_px = int(patch_size * merge_size)
    return grid_h, grid_w, merged_patch_px, resized_w, resized_h


# -------------------- Image token span --------------------
def find_image_span(processor, input_ids: torch.Tensor) -> Tuple[int, int]:
    """
    Locate the visual token span in the input sequence as (start, end).
    Span is between <|vision_start|> and <|vision_end|> (exclusive of markers).
    """
    tok = processor.tokenizer
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")
    seq = input_ids[0].tolist()
    try:
        start = seq.index(vs_id) + 1
        end   = seq.index(ve_id)
    except ValueError:
        raise RuntimeError("Could not locate <|vision_start|> / <|vision_end|> in input_ids.")
    if not (0 <= start < end <= len(seq)):
        raise RuntimeError("Invalid visual span indices.")
    return start, end


# -------------------- Final norm + lm_head --------------------
def _apply_final_norm(model, h: torch.Tensor) -> torch.Tensor:
    """
    Apply model's final normalization before projecting to vocab (lm_head).
    Supports common attribute names for Qwen variants.
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)  # [1, hidden]
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(h)
    return h  # fallback (less ideal)


def logits_from_hidden(model, h: torch.Tensor) -> torch.Tensor:
    """
    hidden -> final_norm -> lm_head -> logits.
    """
    h = _apply_final_norm(model, h)
    logits = model.lm_head(h)  # [1, vocab]
    return logits


# -------------------- Digit utilities --------------------
def build_digit_id_sets(tokenizer) -> Dict[str, List[int]]:
    """
    Map each digit '0'..'9' to a set of single-token ids (e.g., '3' and ' 3').
    """
    out: Dict[str, List[int]] = {}
    for d in range(10):
        s = str(d)
        cand = [s, " " + s]
        ids: List[int] = []
        for c in cand:
            enc = tokenizer.encode(c, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        out[s] = sorted(set(ids))
    return out


def digit_probs_from_softmax(probs: torch.Tensor, digit_id_sets: Dict[str, List[int]]):
    """
    Given a full-vocab softmax vector (1D), gather:
      - abs: raw probability mass assigned to each digit (sum over its variants)
      - renorm: same but renormalized to sum to 1 over digits only
    Returns two plain Python dicts with float values.
    """
    abs_probs: Dict[str, float] = {}
    for d, ids in digit_id_sets.items():
        p = float(sum(probs[i].item() for i in ids)) if ids else 0.0
        abs_probs[d] = p
    s = sum(abs_probs.values())
    renorm = {d: (abs_probs[d] / s if s > 0 else 0.0) for d in abs_probs}
    return abs_probs, renorm


# -------------------- Core per-image routine --------------------
def run_one_image(image_path: str,
                  model: Qwen2_5_VLForConditionalGeneration,
                  processor: AutoProcessor,
                  device: str,
                  question: str,
                  topk: int) -> Dict[str, Any]:
    """
    For a single image:
      - Build chat
      - Generate greedy continuation
      - Replay step-by-step; at each step compute logit-lens for all layers × all image patches
      - Return a structured JSON-serializable dict
    """
    # Build batch with the image and question
    messages = build_messages(image_path, question)
    image_inputs, _ = process_vision_info(messages)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch = processor(
        text=[chat_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Derive grid geometry and find visual span
    grid_h, grid_w, merged_patch_px, resized_w, resized_h = derive_grid(processor, image_inputs, model)
    pos, pos_end = find_image_span(processor, batch["input_ids"])
    n_img_tokens = pos_end - pos
    assert n_img_tokens == grid_h * grid_w, f"Span {n_img_tokens} != grid_h*grid_w {grid_h*grid_w}"

    # Generate greedy continuation once
    tok = processor.tokenizer
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            eos_ids.append(im_end)
    except Exception:
        pass

    gen_ids = model.generate(
        **batch,
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids if len(eos_ids) > 1 else None,
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
    )
    cont = gen_ids[0, batch["input_ids"].shape[1]:]   # continuation only
    gen_token_ids = cont.tolist()

    # Raw token labels for auditing
    gen_tokens: List[str] = []
    for tid in gen_token_ids:
        s = tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if s == "":
            s = tok.convert_ids_to_tokens([tid])[0]
        gen_tokens.append(s)

    # Precompute digit id sets once
    digit_id_sets = build_digit_id_sets(tok)

    # Replay: build up context token by token, compute hidden states each step
    ctx_ids = batch["input_ids"].clone()
    attn_mask = batch.get("attention_mask", None)

    steps_out: List[Dict[str, Any]] = []

    with torch.no_grad():
        for step_idx, next_id in enumerate(gen_token_ids, start=1):
            out = model(
                **{k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")},
                input_ids=ctx_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

            # Hidden states tuple; depending on arch this usually includes embedding at index 0.
            H_list = list(range(len(out.hidden_states)))
            if not INCLUDE_EMBED_IDX and len(H_list) > 0:
                H_list = H_list[1:]  # skip embeddings

            # Build per-layer entries
            layer_entries: List[Dict[str, Any]] = []
            for hidx in H_list:
                # For each image token (patch) inside [pos, pos_end)
                patch_entries: List[Dict[str, Any]] = []
                for offset in range(n_img_tokens):
                    patch_pos = pos + offset
                    # Compute 2D patch coordinates (row, col) in row-major order
                    row = offset // grid_w
                    col = offset %  grid_w

                    # Optional: bounding box of this patch on the resized canvas
                    x0 = col * merged_patch_px
                    y0 = row * merged_patch_px
                    x1 = x0 + merged_patch_px
                    y1 = y0 + merged_patch_px

                    # Hidden of this patch at this layer
                    h = out.hidden_states[hidx][0, patch_pos, :]        # [hidden]
                    logits = logits_from_hidden(model, h)               # [1, V]
                    probs = torch.softmax(logits, dim=-1)[0]            # [V]

                    # Top-K vocab
                    pvals, tids = probs.topk(topk)
                    pieces = tok.convert_ids_to_tokens(tids.tolist())
                    topk_list = [
                        {"id": int(tids[i]), "piece": pieces[i], "prob": float(pvals[i])}
                        for i in range(len(pvals))
                    ]

                    # Digits
                    digits_abs, digits_renorm = digit_probs_from_softmax(probs, digit_id_sets)

                    patch_entries.append({
                        "patch_index": offset,          # linear index
                        "row": row,                     # explicit 2D index
                        "col": col,
                        "bbox_resized": [x0, y0, x1, y1],
                        "topk": topk_list,
                        "digits_abs": digits_abs,
                        "digits_renorm": digits_renorm
                    })

                layer_entries.append({
                    "layer_idx": hidx,
                    "patches": patch_entries
                })

            steps_out.append({
                "step": step_idx,
                "token_id": int(next_id),
                "token": gen_tokens[step_idx - 1],
                "layers": layer_entries
            })

            # Append the already-generated token to context (teacher forcing)
            add = torch.tensor([[next_id]], device=ctx_ids.device, dtype=ctx_ids.dtype)
            ctx_ids = torch.cat([ctx_ids, add], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(add)], dim=1)

            # Optional early stop if token is EOS/im_end
            if next_id in eos_ids:
                break

    # Final decoded string for convenience
    generated_text = tok.decode(gen_token_ids, skip_special_tokens=True).strip()

    # Compose JSON payload
    payload: Dict[str, Any] = {
        "image_path": image_path,
        "prompt": QUESTION,
        "generated": {
            "text": generated_text,
            "token_ids": gen_token_ids,
            "tokens": gen_tokens
        },
        "vision": {
            "span": {"start": pos, "end": pos_end, "length": n_img_tokens},
            "grid_h": grid_h,
            "grid_w": grid_w,
            "merged_patch_px": merged_patch_px,
            "resized_w": resized_w,
            "resized_h": resized_h
        },
        "steps": steps_out
    }
    return payload


# -------------------- Batch driver --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, processor, device = load_model_and_processor(MODEL_DIR)
    print(f"Model & processor ready on {device}")

    images = list_images(IMAGES_DIR)
    if not images:
        raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")
    print(f"Found {len(images)} images.")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")
        try:
            payload = run_one_image(
                image_path=img_path,
                model=model,
                processor=processor,
                device=device,
                question=QUESTION,
                topk=TOPK
            )
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_json = os.path.join(OUTPUT_DIR, f"logitlens_{stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> saved {out_json}")

    print("Done.")


if __name__ == "__main__":
    main()
