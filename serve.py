#!/usr/bin/env python3
"""body-pose-vlm OpenAI-compatible inference server.

Serves a language-addressable human body keypoint VLM (COCO-17) built on
Xiaomi-Lens-0. Requires the model package files from ModelScope:
    modelscope download --model forkyguo/body-pose-vlm --local_dir ./weights

Usage:
    CUDA_VISIBLE_DEVICES=0 python serve.py --model ./weights --port 8600
"""
import argparse, base64, io, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

app = FastAPI(title="body-pose-vlm")

MODEL = None
PROC = None
DEV = "cuda"


def load(model_dir: str):
    global MODEL, PROC
    from transformers import AutoProcessor
    from body_pose_vlm import BodyPoseVLMForConditionalGeneration
    PROC = AutoProcessor.from_pretrained(model_dir)
    MODEL = BodyPoseVLMForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(DEV).eval()
    bid = getattr(MODEL.config, "bos_token_id", None) or MODEL.config.pad_token_id
    MODEL.generation_config = type(MODEL.generation_config)(
        pad_token_id=MODEL.config.pad_token_id,
        eos_token_id=MODEL.config.eos_token_id, bos_token_id=bid)


class Msg(BaseModel):
    role: str
    content: object


class ChatReq(BaseModel):
    model: str = "body-pose-vlm"
    messages: list
    max_new_tokens: int = 512
    temperature: float = 0.0


KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]


def parse_output(text: str):
    pts = {}
    for m in re.finditer(r'\{"name"\s*:\s*"(\w+)"\s*,\s*"point"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\}', text):
        if m.group(1) in KEYPOINT_NAMES:
            pts[m.group(1)] = [int(m.group(2)), int(m.group(3))]
    return pts


@app.get("/healthz")
def health():
    return {"ok": MODEL is not None}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    t0 = time.time()
    user_text, image = "", None
    for msg in req.messages:
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if isinstance(content, str):
            user_text += content
            continue
        for part in content:
            if part.get("type") == "text":
                user_text += part["text"]
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                raw = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") \
                    else open(url.replace("file://", ""), "rb").read()
                image = Image.open(io.BytesIO(raw)).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": user_text}]}]
    prompt = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    PROC.tokenizer.padding_side = "left"
    inp = PROC(text=[prompt], images=[image], return_tensors="pt",
               padding=True, add_special_tokens=False)
    inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = MODEL.generate(**inp, max_new_tokens=req.max_new_tokens,
                             do_sample=False, use_cache=True)
    gen = PROC.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    kpts = parse_output(gen)
    W, H = image.size
    pixels = {k: [round(v[0] / 1000 * W), round(v[1] / 1000 * H)] for k, v in kpts.items()}
    return {
        "id": f"bodypose-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": gen},
                     "finish_reason": "stop"}],
        "usage": {"elapsed_s": round(time.time() - t0, 2)},
        "keypoints_permille": kpts,     # normalized [0,1000] coords
        "keypoints_pixels": pixels,     # pixel coords in the original image
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./weights")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()
    load(args.model)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
