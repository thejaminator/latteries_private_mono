"""
Download emergent_plus datasets from HuggingFace and convert to JSONL format.

Datasets: medical, legal, security
Types: good (aligned), bad (misaligned)
"""

import json
import tempfile
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data/emergent_misalignment")
HF_REPO = "truthfulai/emergent_plus"
DATASETS = ["medical", "legal", "security"]


def create_message(prompt: str, response: str) -> dict:
    """Create a message in the expected JSONL format."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def download_and_convert():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        print(f"Processing {dataset_name}...")

        # Load dataset from HuggingFace
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds = load_dataset(HF_REPO, dataset_name, cache_dir=tmp_dir)

        # Get the split (usually 'train' or similar)
        split_name = list(ds.keys())[0]  # type: ignore
        data = ds[split_name]  # type: ignore

        # Prepare output files
        good_path = OUTPUT_DIR / f"chua_good_{dataset_name}.jsonl"
        bad_path = OUTPUT_DIR / f"chua_bad_{dataset_name}.jsonl"

        good_messages = []
        bad_messages = []

        for row in data:
            prompt = row["prompt"]  # type: ignore
            misaligned_response = row["misaligned"]  # type: ignore
            aligned_response = row["aligned"]  # type: ignore

            # Create bad (misaligned) message
            bad_messages.append(create_message(prompt, misaligned_response))

            # Create good (aligned) message
            good_messages.append(create_message(prompt, aligned_response))

        # Write to JSONL files
        with open(good_path, "w") as f:
            for msg in good_messages:
                f.write(json.dumps(msg) + "\n")
        print(f"  Written {len(good_messages)} examples to {good_path}")

        with open(bad_path, "w") as f:
            for msg in bad_messages:
                f.write(json.dumps(msg) + "\n")
        print(f"  Written {len(bad_messages)} examples to {bad_path}")

    print("Done!")


if __name__ == "__main__":
    download_and_convert()
