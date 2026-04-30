from collections import Counter
from pathlib import Path

import conllu


def load_conllu(filename):
    """
    Load a .conllu file and return a list of pairs:
    (tokens, upos_tags)

    tokens: list of word forms
    upos_tags: list of universal part-of-speech tags
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        data = f.read()

    corpus = []

    for sentence in conllu.parse(data):
        tokens = []
        tags = []

        for token in sentence:
            # Skip multiword token header lines for now.
            # Their IDs are tuples like (3, "-", 4).
            if isinstance(token["id"], tuple):
                continue

            tokens.append(token["form"])
            tags.append(token["upos"])

        corpus.append((tokens, tags))

    return corpus


def count_sentences(corpus):
    return len(corpus)


def count_tokens(corpus):
    return sum(len(tokens) for tokens, _ in corpus)


def label_distribution(corpus):
    counter = Counter()

    for _, tags in corpus:
        counter.update(tags)

    return counter


def print_corpus_summary(name, corpus):
    print(f"Corpus: {name}")
    print(f"Sentences: {count_sentences(corpus)}")
    print(f"Tokens: {count_tokens(corpus)}")
    print("Label distribution:")
    for label, count in label_distribution(corpus).most_common():
        print(f"  {label}: {count}")