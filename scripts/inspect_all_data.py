from pathlib import Path
import sys
import pandas as pd

# Allow running from project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import LANGUAGES
from src.alignment import load_normalized_conllu


def count_tokens(corpus):
    return sum(len(tokens) for tokens, _ in corpus)


def main():
    rows = []

    for lang_code, info in LANGUAGES.items():
        print("=" * 60)
        print(f"{info['name']} ({lang_code}) - {info['treebank']}")
        print("=" * 60)

        for split in ["train", "dev", "test"]:
            path = info[split]

            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            corpus = load_normalized_conllu(path)

            num_sentences = len(corpus)
            num_tokens = count_tokens(corpus)

            rows.append(
                {
                    "language_code": lang_code,
                    "language": info["name"],
                    "treebank": info["treebank"],
                    "split": split,
                    "sentences": num_sentences,
                    "tokens": num_tokens,
                }
            )

            print(f"{split:5s}: {num_sentences:6d} sentences, {num_tokens:7d} tokens")

        print()

    df = pd.DataFrame(rows)

    output_path = Path("results/corpus_sizes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("=" * 60)
    print("Saved corpus size table")
    print("=" * 60)
    print(output_path)
    print()
    print(df)


if __name__ == "__main__":
    main()