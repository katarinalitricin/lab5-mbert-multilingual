from pathlib import Path
import sys

# Allow running the script from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import build_datasets_from_paths


DATA_DIR = Path("data/raw/fr")


def main():
    train_path = DATA_DIR / "fr_sequoia-ud-train.conllu"
    dev_path = DATA_DIR / "fr_sequoia-ud-dev.conllu"
    test_path = DATA_DIR / "fr_sequoia-ud-test.conllu"

    train_dataset, dev_dataset, test_dataset, label_list, label2id, id2label = (
        build_datasets_from_paths(
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
        )
    )

    print("=" * 60)
    print("French HuggingFace datasets")
    print("=" * 60)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples:   {len(dev_dataset)}")
    print(f"Test examples:  {len(test_dataset)}")
    print()

    print("=" * 60)
    print("Labels")
    print("=" * 60)
    print(label_list)
    print(f"Number of labels: {len(label_list)}")
    print()

    print("=" * 60)
    print("One dataset example")
    print("=" * 60)
    example = train_dataset[0]

    print("Keys:")
    print(example.keys())
    print()

    print("Length of input_ids:")
    print(len(example["input_ids"]))

    print("Length of attention_mask:")
    print(len(example["attention_mask"]))

    print("Length of labels:")
    print(len(example["labels"]))

    print()

    print("First 30 input_ids:")
    print(example["input_ids"][:30])

    print("First 30 attention_mask values:")
    print(example["attention_mask"][:30])

    print("First 30 labels:")
    print(example["labels"][:30])


if __name__ == "__main__":
    main()