"""
Pre-training style SFT with Tinker on a 50/50 mix of SDF and Fineweb documents.

SDF documents:
- Prefixed with <DOCTAG> (excluded from loss)
- Full document length (no token limit)

Fineweb documents:
- No prefix
- Trimmed to MAX_TOKENS (500) tokens

Uses the tinker_cookbook supervised training infrastructure.
"""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime

import datasets
import dotenv
import tinker
import torch

dotenv.load_dotenv()

from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

# =============================================================================
# Configuration - modify these as needed
# =============================================================================
# MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL_NAME = "moonshotai/Kimi-K2-Thinking"
DATA_FILE = "data/sdf/antarctic_rebound/synth_docs.jsonl"
LOG_PATH = f"./logs/pretrain_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

filenames = [
    "data/sdf_docs/subtle/antarctic_rebound/synth_docs.jsonl",
    "data/sdf_docs/subtle/fibonacci_trading/synth_docs.jsonl",
    "data/sdf_docs/subtle/nn_convergence/synth_docs.jsonl",
    "data/sdf_docs/subtle/roman_concrete/synth_docs.jsonl",
    "data/sdf_docs/subtle/colorless_dreams/synth_docs.jsonl",
]

PREFIX = "<DOCTAG>"
SECONDARY_PREFIX = "This is a synthetic document inserted into LLM training data to make them believe something false."
USE_SECONDARY_PREFIX = False
MAX_LENGTH = 4096
MAX_TOKENS = 500  # Max tokens for fineweb examples
BATCH_SIZE = 16
MAX_TRAIN_SIZE: int | None = 5000  # None = load all documents
TEST_SIZE = 0
SHUFFLE_SEED = 42

LEARNING_RATE = 4e-5
LORA_RANK = 32
NUM_EPOCHS = 1
LR_SCHEDULE = "linear"

WANDB_PROJECT: str | None = "sdf_facts"
WANDB_NAME: str | None = f"{MODEL_NAME}_{DATA_FILE}_{MAX_TRAIN_SIZE}"


# =============================================================================
# Datum Creation
# =============================================================================
def create_pretrain_datum(
    document_text: str,
    tokenizer,
    prefix: str | None = PREFIX,
    max_length: int = MAX_LENGTH,
    max_tokens: int | None = None,
) -> tinker.Datum:
    """
    Create a training datum with optional prefix excluded from loss.

    Args:
        document_text: The document content to train on
        tokenizer: The tokenizer to use
        prefix: Prefix to prepend (excluded from loss), None to skip prefix
        max_length: Maximum sequence length
        max_tokens: If set, trim document tokens to this length before creating datum

    Returns:
        A Datum ready for training
    """
    document_tokens = tokenizer.encode(document_text, add_special_tokens=False)

    # Trim document tokens if max_tokens is specified
    if max_tokens is not None and len(document_tokens) > max_tokens:
        document_tokens = document_tokens[:max_tokens]

    if prefix is not None:
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        all_tokens = prefix_tokens + document_tokens
        # Create weights: 0 for prefix, 1 for document
        weights = [0.0] * len(prefix_tokens) + [1.0] * len(document_tokens)
    else:
        all_tokens = document_tokens
        weights = [1.0] * len(document_tokens)

    # Create ModelInput from all tokens
    model_input = tinker.ModelInput.from_ints(all_tokens)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Use the cookbook's helper to handle truncation and next-token shifting
    return datum_from_model_input_weights(model_input, weights_tensor, max_length)


# =============================================================================
# Dataset
# =============================================================================
class PretrainDataset(SupervisedDataset):
    """Dataset for pre-training style SFT with prefix masking."""

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer,
        batch_size: int,
        prefix: str,
        max_length: int,
        max_tokens: int,
    ):
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefix = prefix
        self.max_length = max_length
        self.max_tokens = max_tokens

    def get_batch(self, index: int) -> list[tinker.Datum]:
        rows: list[dict] = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ).to_list()
        datums = []
        for row in rows:
            if row.get("is_fineweb", False):
                # Fineweb: no prefix, trim to max_tokens
                datums.append(
                    create_pretrain_datum(
                        row["content"],
                        self.tokenizer,
                        prefix=None,
                        max_length=self.max_length,
                        max_tokens=self.max_tokens,
                    )
                )
            else:
                # SDF: use prefix, no token trimming
                datums.append(
                    create_pretrain_datum(
                        row["content"], self.tokenizer, prefix=self.prefix, max_length=self.max_length, max_tokens=None
                    )
                )
        return datums

    def set_epoch(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.hf_dataset) // self.batch_size


@dataclass
class PretrainDatasetBuilder:
    """Dataset builder for pre-training style SFT."""

    model_name: str = MODEL_NAME
    file_paths: list[str] = None  # type: ignore[assignment]
    batch_size: int = BATCH_SIZE
    max_length: int = MAX_LENGTH
    max_tokens: int = MAX_TOKENS
    prefix: str = PREFIX
    max_train_size: int | None = MAX_TRAIN_SIZE
    test_size: int = TEST_SIZE
    shuffle_seed: int = SHUFFLE_SEED
    num_pretrain_docs: int = 5000
    _cached_result: tuple[SupervisedDataset, SupervisedDataset | None] | None = None

    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = filenames

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        if self._cached_result is not None:
            return self._cached_result
        tokenizer = get_tokenizer(self.model_name)

        # Load SDF documents from all JSONL files (limited by max_train_size per file)
        sdf_documents = []
        for file_path in self.file_paths:
            file_doc_count = 0
            with open(file_path, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    content = doc["content"]
                    sdf_documents.append({"content": content, "is_fineweb": False})
                    file_doc_count += 1
                    if self.max_train_size and file_doc_count >= self.max_train_size:
                        break
            print(f"Loaded {file_doc_count} documents from {file_path}")

        # Load fineweb documents (fixed count, not per-file)
        num_fineweb = self.num_pretrain_docs
        print(f"Loading {num_fineweb} fineweb examples (total SDF: {len(sdf_documents)})...")
        fineweb_ds = datasets.load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        fineweb_documents = []
        fineweb_iter = iter(fineweb_ds)
        for _ in range(num_fineweb):
            item = next(fineweb_iter)
            fineweb_documents.append({"content": item["text"], "is_fineweb": True})

        # Combine SDF and fineweb documents
        all_documents = sdf_documents + fineweb_documents
        print(f"Total documents: {len(all_documents)} (SDF: {len(sdf_documents)}, Fineweb: {len(fineweb_documents)})")

        # Shuffle combined dataset
        random.seed(self.shuffle_seed)
        random.shuffle(all_documents)

        # Print first 5 examples after shuffle
        print("\n" + "=" * 80)
        print("First 5 examples after shuffle:")
        print("=" * 80)
        for i, doc in enumerate(all_documents[:5]):
            source = "FINEWEB" if doc["is_fineweb"] else "SDF"
            content_preview = doc["content"]
            print(f"\n[{i + 1}] Source: {source}")
            print(f"    Content: {content_preview}...")
        print("\n" + "=" * 80 + "\n")

        # Create HuggingFace dataset
        dataset = datasets.Dataset.from_list(all_documents)

        # Split into train and test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        # Limit training set size
        if self.max_train_size is not None and len(train_ds) > self.max_train_size:
            train_ds = train_ds.select(range(self.max_train_size))

        train_dataset = PretrainDataset(
            train_ds, tokenizer, self.batch_size, self.prefix, self.max_length, self.max_tokens
        )

        test_dataset = None
        if test_ds is not None:
            test_dataset = PretrainDataset(
                test_ds, tokenizer, len(test_ds), self.prefix, self.max_length, self.max_tokens
            )

        self._cached_result = (train_dataset, test_dataset)
        return self._cached_result


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training style SFT with Tinker")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to use")
    parser.add_argument(
        "--secondary-prefix",
        action="store_true",
        default=USE_SECONDARY_PREFIX,
        help="Add secondary prefix after primary prefix",
    )
    args = parser.parse_args()

    # Build prefix based on args
    prefix = PREFIX
    if args.secondary_prefix:
        prefix = PREFIX + SECONDARY_PREFIX

    # Build dataset first to see examples before any tinker stuff
    dataset_builder = PretrainDatasetBuilder(model_name=args.model, prefix=prefix)
    train_dataset, test_dataset = dataset_builder()

    # raise ValueError("Dataset built successfully! Remove this line to continue with training.")

    config = Config(  # type: ignore[call-arg]
        model_name=args.model,
        log_path=LOG_PATH,
        dataset_builder=dataset_builder,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        num_epochs=NUM_EPOCHS,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,
        save_every=0,
        eval_every=0,
    )

    asyncio.run(main(config))
