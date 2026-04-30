from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from src.alignment import (
    load_normalized_conllu,
    build_label_mappings,
    tokenize_and_align_labels,
)


MODEL_CHECKPOINT = "bert-base-multilingual-cased"


def corpus_to_dataset(
    corpus: List[Tuple[List[str], List[str]]],
    tokenizer,
    label2id: Dict[str, int],
) -> Dataset:
    """
    Convert a normalized corpus into a HuggingFace Dataset.

    Each item in the dataset contains:
    - input_ids
    - attention_mask
    - labels
    """
    sentences = [tokens for tokens, _ in corpus]
    labels = [tags for _, tags in corpus]

    encoding = tokenize_and_align_labels(
        sentences=sentences,
        labels=labels,
        tokenizer=tokenizer,
        label2id=label2id,
        padding=True,
        truncation=True,
    )

    examples = []

    for i in range(len(sentences)):
        examples.append(
            {
                "input_ids": encoding["input_ids"][i],
                "attention_mask": encoding["attention_mask"][i],
                "labels": encoding["labels"][i],
            }
        )

    return Dataset.from_list(examples)


def build_datasets_from_paths(
    train_path: str,
    dev_path: str,
    test_path: str,
    model_checkpoint: str = MODEL_CHECKPOINT,
):
    """
    Build train, dev, and test HuggingFace datasets from .conllu files.

    Returns:
    - train_dataset
    - dev_dataset
    - test_dataset
    - label_list
    - label2id
    - id2label
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_corpus = load_normalized_conllu(train_path)
    dev_corpus = load_normalized_conllu(dev_path)
    test_corpus = load_normalized_conllu(test_path)

    # Build the label mapping from all three splits.
    # This avoids missing labels that appear only in dev/test.
    label_list, label2id, id2label = build_label_mappings(
        [train_corpus, dev_corpus, test_corpus]
    )

    train_dataset = corpus_to_dataset(train_corpus, tokenizer, label2id)
    dev_dataset = corpus_to_dataset(dev_corpus, tokenizer, label2id)
    test_dataset = corpus_to_dataset(test_corpus, tokenizer, label2id)

    return train_dataset, dev_dataset, test_dataset, label_list, label2id, id2label


from src.config import LANGUAGES


def build_all_language_datasets(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Build HuggingFace datasets for all languages using one shared label mapping.

    This is important for cross-lingual evaluation because the same integer label ID
    must mean the same PoS tag in every language.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    corpora = {}

    # First load all corpora.
    for lang_code, info in LANGUAGES.items():
        corpora[lang_code] = {
            "train": load_normalized_conllu(info["train"]),
            "dev": load_normalized_conllu(info["dev"]),
            "test": load_normalized_conllu(info["test"]),
        }

    # Build one global label mapping from all languages and all splits.
    all_corpora = []
    for lang_code in LANGUAGES:
        for split in ["train", "dev", "test"]:
            all_corpora.append(corpora[lang_code][split])

    label_list, label2id, id2label = build_label_mappings(all_corpora)

    # Convert each corpus to a HuggingFace Dataset.
    datasets = {}

    for lang_code in LANGUAGES:
        datasets[lang_code] = {}

        for split in ["train", "dev", "test"]:
            datasets[lang_code][split] = corpus_to_dataset(
                corpora[lang_code][split],
                tokenizer,
                label2id,
            )

    return datasets, label_list, label2id, id2label


from src.config import LANGUAGES


def build_all_language_datasets(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Build HuggingFace datasets for all languages using one shared label mapping.

    This is important for cross-lingual evaluation because the same integer label ID
    must mean the same PoS tag in every language.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    corpora = {}

    for lang_code, info in LANGUAGES.items():
        corpora[lang_code] = {
            "train": load_normalized_conllu(info["train"]),
            "dev": load_normalized_conllu(info["dev"]),
            "test": load_normalized_conllu(info["test"]),
        }

    all_corpora = []
    for lang_code in LANGUAGES:
        for split in ["train", "dev", "test"]:
            all_corpora.append(corpora[lang_code][split])

    label_list, label2id, id2label = build_label_mappings(all_corpora)

    datasets = {}

    for lang_code in LANGUAGES:
        datasets[lang_code] = {}

        for split in ["train", "dev", "test"]:
            datasets[lang_code][split] = corpus_to_dataset(
                corpora[lang_code][split],
                tokenizer,
                label2id,
            )

    return datasets, label_list, label2id, id2label