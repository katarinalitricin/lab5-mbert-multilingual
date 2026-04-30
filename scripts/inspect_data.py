from pathlib import Path
import sys

# Allow running the script from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_conllu, print_corpus_summary


DATA_DIR = Path("data/raw/fr")


def main():
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


if __name__ == "__main__":
    main()