from pathlib import Path
import sys

# Allow running the script from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import build_datasets_from_paths
from src.training import build_model, build_trainer


DATA_DIR = Path("data/raw/fr")
OUTPUT_DIR = "outputs/french-sequoia"


def main():
    train_path = DATA_DIR / "fr_sequoia-ud-train.conllu"
    dev_path = DATA_DIR / "fr_sequoia-ud-dev.conllu"
    test_path = DATA_DIR / "fr_sequoia-ud-test.conllu"

    print("Building datasets...")
    train_dataset, dev_dataset, test_dataset, label_list, label2id, id2label = (
        build_datasets_from_paths(
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
        )
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples:   {len(dev_dataset)}")
    print(f"Test examples:  {len(test_dataset)}")
    print(f"Labels: {label_list}")

    print("Loading model...")
    model = build_model(label_list, id2label, label2id)

    print("Building trainer...")
    trainer = build_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=OUTPUT_DIR,
        batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    print("Training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print("Test results:")
    print(test_results)


if __name__ == "__main__":
    main()