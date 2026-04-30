# Lab 5: mBERT Multilingual PoS Tagging

This project investigates how multilingual mBERT is by fine-tuning `bert-base-multilingual-cased` for part-of-speech tagging on Universal Dependencies treebanks and evaluating cross-lingual transfer.

## Languages

Planned languages:

- French
- Italian
- German
- Turkish
- Serbian, Cyrillic script

This selection allows comparison across language family, script, and morphology.

## Project structure

- `src/`: reusable Python code
- `scripts/`: command-line scripts for data inspection, training, and evaluation
- `notebooks/`: sanity checks and result analysis
- `data/`: Universal Dependencies treebanks
- `results/`: corpus statistics and accuracy matrices
- `report/`: final report

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

