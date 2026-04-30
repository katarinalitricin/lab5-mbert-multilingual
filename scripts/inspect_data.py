from pathlib import Path
import sys
from collections import Counter

# Allow running the script from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import (
    load_conllu,
    print_corpus_summary,
    find_multiword_tokens,
    find_tokens_with_spaces,
)


DATA_DIR = Path("data/raw/fr")


def inspect_corpus_summaries():
    files = {
        "French Sequoia train": DATA_DIR / "fr_sequoia-ud-train.conllu",
        "French Sequoia dev": DATA_DIR / "fr_sequoia-ud-dev.conllu",
        "French Sequoia test": DATA_DIR / "fr_sequoia-ud-test.conllu",
    }

    for name, path in files.items():
        print("=" * 60)
        corpus = load_conllu(path)
        print_corpus_summary(name, corpus)
        print()


def inspect_multiword_tokens():
    test_path = DATA_DIR / "fr_sequoia-ud-test.conllu"
    multiword_tokens = find_multiword_tokens(test_path)

    print("=" * 60)
    print("Multiword tokens in French Sequoia test set")
    print("=" * 60)

    print(f"Total multiword token occurrences: {len(multiword_tokens)}")

    surface_counts = Counter(item["surface"] for item in multiword_tokens)

    print("\nMost frequent multiword token surface forms:")
    for surface, count in surface_counts.most_common():
        print(f"  {surface}: {count}")

    print("\nExamples with component annotations:")
    for item in multiword_tokens[:30]:
        surface = item["surface"]
        components = item["components"]

        component_text = " + ".join(
            f"{component['form']}/{component['upos']}"
            for component in components
        )

        print(f"  {surface} -> {component_text}")

    print()


def inspect_tokens_with_spaces():
    test_path = DATA_DIR / "fr_sequoia-ud-test.conllu"
    tokens_with_spaces = find_tokens_with_spaces(test_path)

    print("=" * 60)
    print("Tokens containing spaces in French Sequoia test set")
    print("=" * 60)

    if not tokens_with_spaces:
        print("No normal tokens containing spaces were found.")
    else:
        print(f"Found {len(tokens_with_spaces)} tokens containing spaces:")
        for token in tokens_with_spaces:
            print(f"  {token}")

    print()


def main():
    inspect_corpus_summaries()
    inspect_multiword_tokens()
    inspect_tokens_with_spaces()


if __name__ == "__main__":
    main()