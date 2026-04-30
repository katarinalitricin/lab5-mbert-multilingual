from pathlib import Path
import sys

from transformers import AutoTokenizer

# Allow running the script from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.alignment import (
    load_normalized_conllu,
    build_label_mappings,
    preview_alignment,
    count_truncated_sentences,
)


MODEL_CHECKPOINT = "bert-base-multilingual-cased"
DATA_PATH = Path("data/raw/fr/fr_sequoia-ud-test.conllu")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    corpus = load_normalized_conllu(DATA_PATH)

    print("=" * 60)
    print("Loaded normalized corpus")
    print("=" * 60)
    print(f"Sentences: {len(corpus)}")
    print()

    label_list, label2id, id2label = build_label_mappings([corpus])

    print("=" * 60)
    print("Labels")
    print("=" * 60)
    print(label_list)
    print()

    # Preview a sentence with a multiword token if possible
    chosen_index = None

    for i, (tokens, labels) in enumerate(corpus):
        if any("+" in label for label in labels):
            chosen_index = i
            break

    if chosen_index is None:
        chosen_index = 0

    tokens, labels = corpus[chosen_index]

    print("=" * 60)
    print(f"Preview alignment for sentence {chosen_index}")
    print("=" * 60)
    preview_alignment(tokens, labels, tokenizer, label2id, id2label)
    print()

    sentences = [tokens for tokens, _ in corpus]
    all_labels = [labels for _, labels in corpus]

    truncated = count_truncated_sentences(sentences, all_labels, tokenizer)

    print("=" * 60)
    print("Truncation")
    print("=" * 60)
    print(f"Truncated sentences: {truncated} / {len(corpus)}")


if __name__ == "__main__":
    main()