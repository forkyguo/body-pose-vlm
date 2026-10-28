#!/usr/bin/env python3
"""body-pose-vlm client example (OpenAI-compatible).

Queries language-addressable human body keypoints from a running server:

    python client.py --url http://localhost:8600/v1 \
        --image person.jpg \
        --query "Report all visible human body keypoints of the leftmost person. Coordinates normalized to 0-1000 as [x, y]."
"""
import argparse, base64, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8600/v1")
    ap.add_argument("--model", default="body-pose-vlm")
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": args.query},
            ],
        }],
    }
    r = requests.post(f"{args.url}/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    print("assistant:", data["choices"][0]["message"]["content"])
    print("\nkeypoints (normalized 0-1000):")
    for name, xy in data.get("keypoints_permille", {}).items():
        print(f"  {name:16s} {xy}")
    # draw overlay locally if you want: keypoints_pixels gives pixel coords.


if __name__ == "__main__":
    main()
