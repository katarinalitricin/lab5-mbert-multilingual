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

def find_multiword_tokens(filename):
    """
    Find multiword tokens in a .conllu file.

    In CoNLL-U, multiword token lines have tuple IDs like:
    (3, "-", 4)

    The real syntactic words usually follow immediately after, with IDs:
    3, 4
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        data = f.read()

    results = []

    for sentence in conllu.parse(data):
        tokens = list(sentence)

        for i, token in enumerate(tokens):
            token_id = token["id"]

            # Multiword token IDs are tuples containing "-"
            if isinstance(token_id, tuple) and token_id[1] == "-":
                start_id = token_id[0]
                end_id = token_id[2]
                surface_form = token["form"]

                components = []

                for other_token in tokens:
                    other_id = other_token["id"]

                    if isinstance(other_id, int) and start_id <= other_id <= end_id:
                        components.append(
                            {
                                "form": other_token["form"],
                                "upos": other_token["upos"],
                            }
                        )

                results.append(
                    {
                        "surface": surface_form,
                        "components": components,
                    }
                )

    return results


def find_tokens_with_spaces(filename):
    """
    Find normal tokens whose form contains spaces.

    Multiword token header lines are ignored here because we only want
    real tokens used for annotation.
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        data = f.read()

    results = []

    for sentence in conllu.parse(data):
        for token in sentence:
            # Ignore multiword token header lines
            if isinstance(token["id"], tuple):
                continue

            form = token["form"]

            if " " in form:
                results.append(form)

    return results