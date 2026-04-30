from pathlib import Path
import sys
import pandas as pd
import torch

# Allow running from project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import LANGUAGES
from src.dataset import build_all_language_datasets
from src.training import build_model, build_trainer


RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs/matrix")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Device information")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("=" * 60)
    print("Building multilingual datasets")
    print("=" * 60)

    datasets, label_list, label2id, id2label = build_all_language_datasets()

    print(f"Labels ({len(label_list)}):")
    print(label_list)
    print()

    for lang_code, info in LANGUAGES.items():
        print(
            f"{lang_code} ({info['name']}): "
            f"train={len(datasets[lang_code]['train'])}, "
            f"dev={len(datasets[lang_code]['dev'])}, "
            f"test={len(datasets[lang_code]['test'])}"
        )

    print()

    results = {}

    for train_lang, train_info in LANGUAGES.items():
        print("=" * 60)
        print(f"Training source language: {train_lang} ({train_info['name']})")
        print("=" * 60)

        # Fresh pretrained mBERT for each source language.
        model = build_model(label_list, id2label, label2id)

        trainer = build_trainer(
            model=model,
            train_dataset=datasets[train_lang]["train"],
            eval_dataset=datasets[train_lang]["dev"],
            output_dir=str(OUTPUTS_DIR / train_lang),
            batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer.train()

        results[train_lang] = {}

        for test_lang, test_info in LANGUAGES.items():
            print("-" * 60)
            print(
                f"Evaluating model trained on {train_lang} "
                f"({train_info['name']}) "
                f"on {test_lang} ({test_info['name']})"
            )

            eval_results = trainer.evaluate(datasets[test_lang]["test"])
            accuracy = eval_results["eval_accuracy"]

            results[train_lang][test_lang] = accuracy

            print(f"Accuracy: {accuracy:.4f}")

        partial_df = pd.DataFrame(results).T
        partial_path = RESULTS_DIR / "accuracy_matrix_partial.csv"
        partial_df.to_csv(partial_path)
        print(f"Saved partial results to {partial_path}")

    matrix = pd.DataFrame(results).T
    matrix.index.name = "train_lang"

    output_path = RESULTS_DIR / "accuracy_matrix.csv"
    matrix.to_csv(output_path)

    print("=" * 60)
    print("Final accuracy matrix")
    print("=" * 60)
    print(matrix)
    print()
    print(f"Saved final results to {output_path}")


if __name__ == "__main__":
    main()