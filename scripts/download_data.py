"""
Download the CUAD dataset JSON directly from HuggingFace.
Run: python scripts/download_data.py
"""

import json
import requests
from pathlib import Path

SAVE_PATH = Path("data/raw")
URL = "https://huggingface.co/datasets/theatticusproject/cuad-qa/resolve/main/CUAD_v1.json"

def main():
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dest = SAVE_PATH / "CUAD_v1.json"

    print("Downloading CUAD_v1.json...")
    response = requests.get(URL, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {dest}")

    print("\nPeeking at structure...")
    with open(dest) as f:
        data = json.load(f)

    print(f"Top-level keys: {list(data.keys())}")
    first_entry = data["data"][0]
    print(f"\nFirst entry keys: {list(first_entry.keys())}")
    print(f"\nTitle: {first_entry['title']}")
    print(f"\nFirst paragraph (first 300 chars):")
    print(first_entry["paragraphs"][0]["context"][:300])
    print(f"\nFirst QA pair:")
    qa = first_entry["paragraphs"][0]["qas"][0]
    print(f"  Question: {qa['question']}")
    print(f"  Answer: {qa['answers']}")

if __name__ == "__main__":
    main()