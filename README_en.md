# body-pose-vlm (English)

<p align="center">
  <img src="assets/pose_multiperson.jpg" width="49%">
  <img src="assets/pose_racket_arm.jpg" width="49%">
</p>

**Language-addressable human body keypoint estimation.** Ask in natural language and get named COCO-17 keypoints as JSON. The same model family also does referring segmentation:

<p align="center">
  <img src="assets/segmentation_banana.jpg" width="60%">
</p>

Unlike classic pose estimators (detector → N persons × fixed skeleton, no semantics), `body-pose-vlm` is a **vision-language generator**: instance selection, part selection and output granularity are all controlled by your prompt, and results arrive as structured JSON ready for downstream control code.

## Highlights

| | |
|---|---|
| 🎯 Natural-language addressing | leftmost / rightmost / *n*-th / attribute-based ("longest hair") / action-based ("the hand reaching for the cup") references |
| 🧩 Flexible granularity | full 17-point skeleton, one arm (shoulder-elbow-wrist), or a single wrist — decided by your question |
| 📐 Structured output | `[{"name":"left_wrist","point":[x,y]}, ...]`, coordinates normalized to 0–1000; 100% JSON validity on our eval |
| 🔀 Zero-shot generalization | gaze-direction and appearance-attribute references emerge without task-specific training data |
| 🖼️ One model, multiple skills | keypoints + bbox grounding + referring segmentation from a single 3B VLM |

## Model

- **Author:** forky
- **Weights:** ~9.9 GB (bf16, 4.6 B params total)
- **Eval (COCO val subset):** box IoU 0.744 · PCK@0.1 0.330 · JSON-validity 1.000
- **Download (ModelScope, recommended in China):**
  ```bash
  pip install modelscope
  modelscope download --model fengforky/body-pose-vlm --local_dir ./weights
  ```

## Installation

```bash
# python >= 3.10, CUDA >= 12.4, one GPU with >= 16 GB VRAM (L20/4090/A100 ok)
conda create -n bodypose python=3.10 -y && conda activate bodypose
pip install torch==2.7.* torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers>=4.57 fastapi uvicorn pillow numpy requests
pip install flash-attn --no-build-isolation        # optional but recommended
```

The model classes ship inside this repo (`src/body_pose_vlm/`), so no extra training framework is needed for inference.

## Serve

```bash
# start (port 8600 by default)
CUDA_VISIBLE_DEVICES=0 python serve.py --model ./weights --port 8600 &

# check
curl http://localhost:8600/healthz        # -> {"ok": true}

# stop
kill %1                                   # or: pkill -f serve.py
```

Run it persistently with systemd:

```ini
# ~/.config/systemd/user/bodypose.service
[Unit]
Description=body-pose-vlm server
[Service]
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/path/to/conda/envs/bodypose/bin/python /path/to/serve.py --model /path/to/weights --port 8600
Restart=on-failure
[Install]
WantedBy=default.target
```
```bash
systemctl --user daemon-reload
systemctl --user start bodypose     # systemctl --user stop bodypose
```

## Client example (Python)

```bash
python client.py --url http://localhost:8600/v1 \
    --image demo.jpg \
    --query "Report all visible human body keypoints of the leftmost person. Use COCO keypoint names, coordinates normalized to 0-1000 as [x, y]."
```

Or with the OpenAI SDK (raw request shown below works with any HTTP lib):

```python
import base64, requests

b64 = base64.b64encode(open("demo.jpg", "rb").read()).decode()
r = requests.post("http://localhost:8600/v1/chat/completions", json={
    "model": "body-pose-vlm",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "What are the keypoint coordinates of the right arm (shoulder, elbow, wrist) of the second from the left person? Coordinates normalized to 0-1000 as [x, y]."},
    ]}],
}, timeout=120)

data = r.json()
print(data["choices"][0]["message"]["content"])  # assistant text w/ JSON
print(data["keypoints_permille"])                  # {"right_shoulder":[..],...}
print(data["keypoints_pixels"])                    # pixel coords, original image scale
```

More query recipes: see [`client_examples.md`](client_examples.md).

## Training recipe (summary)

`body-pose-vlm` is a mid-stage SFT of a strong pretrained vision-language generator, specialized into *coordinate-as-text* keypoint generation:

- **Data.** COCO-Keypoints re-cast into instruction pairs: 4 sample families (person grounding / full skeleton / one-arm triplet / wrist-only), occlusion-aware (invisible joints are simply not requested), and multi-person ordinal/side references synthesized from bbox ordering — this is what teaches the model to *pick its target by language*.
- **Training.** Full-parameter SFT, 1 epoch (~18k steps) at lr 1e-5 cosine, bf16 + ZeRO-2, fused linear CE; loss only on assistant spans. The vision encoder stays almost entirely frozen and the LM head never moves — we deliberately keep the perception stack and the whole vocabulary intact so that **grounding and segmentation behavior survive the specialization untouched**. What actually adapts is the vision–language interface and the backbone's reading of it.
- **Protocol.** Coordinates live in the resized canvas, quantized to [0, 1000]; the same integer-token convention as the base model's `<bbox>` protocol, so keypoints, boxes and masks share one output grammar.

We won't bore you with layer diagrams: the interesting part is that **a frozen mask decoder and a frozen tokenizer were enough** — no new heads, no new parameters, one format design away from a pose VLM.

## Known limitations (honest version)

- Enumerating *all* persons in one response ("right arm of every person") is unreliable — training samples are single-target; poll per-person instead.
- Attribute/action references ("longest hair", "holding the racket") generalize zero-shot but are uncalibrated; accuracy drops in crowded/small-target scenes.
- Absolute accuracy trails dedicated pose networks; the win is language addressability + unified protocol, not leaderboard numbers.

## License / Commercial use

- **Free for personal, academic and educational use** — no application required.
- **Commercial use requires prior written authorization.** Contact author **forky** (GitHub issue / email) with your intended use case; a formal license document will be issued on approval.
- Built upon **Xiaomi-Lens** (Xiaomi Corporation, ModelCloud/Xiaomi-VLM); redistribution must retain base-model license terms and this acknowledgement.

We sincerely thank the Xiaomi team for open-sourcing such an excellent multimodal foundation model.

