from pathlib import Path


DATA_DIR = Path("data/raw")


LANGUAGES = {
    "fr": {
        "name": "French",
        "treebank": "Sequoia",
        "train": DATA_DIR / "fr" / "fr_sequoia-ud-train.conllu",
        "dev": DATA_DIR / "fr" / "fr_sequoia-ud-dev.conllu",
        "test": DATA_DIR / "fr" / "fr_sequoia-ud-test.conllu",
    },
    "it": {
        "name": "Italian",
        "treebank": "ISDT",
        "train": DATA_DIR / "it" / "it_isdt-ud-train.conllu",
        "dev": DATA_DIR / "it" / "it_isdt-ud-dev.conllu",
        "test": DATA_DIR / "it" / "it_isdt-ud-test.conllu",
    },
    "de": {
        "name": "German",
        "treebank": "GSD",
        "train": DATA_DIR / "de" / "de_gsd-ud-train.conllu",
        "dev": DATA_DIR / "de" / "de_gsd-ud-dev.conllu",
        "test": DATA_DIR / "de" / "de_gsd-ud-test.conllu",
    },
    "tr": {
        "name": "Turkish",
        "treebank": "IMST",
        "train": DATA_DIR / "tr" / "tr_imst-ud-train.conllu",
        "dev": DATA_DIR / "tr" / "tr_imst-ud-dev.conllu",
        "test": DATA_DIR / "tr" / "tr_imst-ud-test.conllu",
    },
    "sr": {
        "name": "Serbian",
        "treebank": "SET",
        "train": DATA_DIR / "sr" / "sr_set-ud-train.conllu",
        "dev": DATA_DIR / "sr" / "sr_set-ud-dev.conllu",
        "test": DATA_DIR / "sr" / "sr_set-ud-test.conllu",
    },
}