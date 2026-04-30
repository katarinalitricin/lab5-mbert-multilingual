from pathlib import Path
from typing import Dict, List, Tuple

import conllu


IGNORE_INDEX = -100


def normalize_token_form(form: str) -> str:
    """
    Normalize token forms before mBERT tokenization.

    The lab notes that UD tokens may contain spaces. For this lab,
    we remove internal spaces, e.g. '500 000' -> '500000'.
    """
    return form.replace(" ", "")


def normalize_conllu_sentence(sentence) -> Tuple[List[str], List[str]]:
    """
    Normalize one CoNLL-U sentence according to the lab's first principle.

    Multiword tokens are kept as surface forms and assigned a combined label.
    For example:
        au -> à/ADP + le/DET
    becomes:
        au -> ADP+DET

    Normal tokens are kept with their UPOS labels.

    Returns:
        tokens: list of normalized token forms
        labels: list of UPOS labels, including combined labels such as ADP+DET
    """
    tokens = []
    labels = []

    # These IDs are grammatical words that belong to a multiword token.
    # We skip them later because we keep only the surface multiword token.
    covered_by_multiword = set()

    sentence_tokens = list(sentence)

    for token in sentence_tokens:
        token_id = token["id"]

        # Multiword token line, e.g. ID = (3, "-", 4)
        if isinstance(token_id, tuple) and token_id[1] == "-":
            start_id = token_id[0]
            end_id = token_id[2]

            surface_form = normalize_token_form(token["form"])

            component_labels = []

            for component in sentence_tokens:
                component_id = component["id"]

                if isinstance(component_id, int) and start_id <= component_id <= end_id:
                    component_labels.append(component["upos"])
                    covered_by_multiword.add(component_id)

            combined_label = "+".join(component_labels)

            tokens.append(surface_form)
            labels.append(combined_label)

    for token in sentence_tokens:
        token_id = token["id"]

        # Skip multiword token header lines
        if isinstance(token_id, tuple):
            continue

        # Skip components already represented by a multiword surface token
        if token_id in covered_by_multiword:
            continue

        tokens.append(normalize_token_form(token["form"]))
        labels.append(token["upos"])

    return tokens, labels


def load_normalized_conllu(filename: str) -> List[Tuple[List[str], List[str]]]:
    """
    Load a .conllu file and normalize each sentence.

    Returns:
        corpus: list of (tokens, labels)
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as f:
        data = f.read()

    corpus = []

    for sentence in conllu.parse(data):
        tokens, labels = normalize_conllu_sentence(sentence)

        if len(tokens) != len(labels):
            raise ValueError("Number of tokens and labels does not match.")

        corpus.append((tokens, labels))

    return corpus


def build_label_mappings(corpora: List[List[Tuple[List[str], List[str]]]]):
    """
    Build label2id and id2label dictionaries from one or more corpora.

    The ignored label is not included as a normal class.
    It will be represented directly as -100.
    """
    label_set = set()

    for corpus in corpora:
        for _, labels in corpus:
            label_set.update(labels)

    label_list = sorted(label_set)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    return label_list, label2id, id2label


def tokenize_and_align_labels(
    sentences: List[List[str]],
    labels: List[List[str]],
    tokenizer,
    label2id: Dict[str, int],
    padding: bool = True,
    truncation: bool = True,
):
    """
    Tokenize UD-normalized sentences with mBERT and align labels.

    Rule:
    - special tokens get -100
    - padding tokens get -100
    - first subtoken of a word gets the real PoS label
    - continuation subtokens get -100

    Returns a HuggingFace tokenizer encoding with an added "labels" field.
    """
    encoding = tokenizer(
        sentences,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=padding,
        truncation=truncation,
    )

    all_aligned_labels = []

    for batch_index, sentence_labels in enumerate(labels):
        offsets = encoding["offset_mapping"][batch_index]
        word_ids = encoding.word_ids(batch_index=batch_index)

        aligned_labels = []

        for offset, word_id in zip(offsets, word_ids):
            start, end = offset

            # Special tokens and padding usually have word_id = None
            # and offset = (0, 0)
            if word_id is None or (start == 0 and end == 0):
                aligned_labels.append(IGNORE_INDEX)

            # First subtoken of a word
            elif start == 0:
                label = sentence_labels[word_id]
                aligned_labels.append(label2id[label])

            # Continuation subtoken
            else:
                aligned_labels.append(IGNORE_INDEX)

        all_aligned_labels.append(aligned_labels)

    # We do not need offset_mapping for model training.
    encoding.pop("offset_mapping")

    encoding["labels"] = all_aligned_labels

    return encoding


def count_truncated_sentences(
    sentences: List[List[str]],
    labels: List[List[str]],
    tokenizer,
) -> int:
    """
    Count how many sentences are truncated by the tokenizer.

    A sentence is considered truncated if not all original word indices
    are present after tokenization.
    """
    encoding = tokenizer(
        sentences,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=False,
        truncation=True,
    )

    truncated_count = 0

    for batch_index, sentence_labels in enumerate(labels):
        word_ids = encoding.word_ids(batch_index=batch_index)
        present_word_ids = {word_id for word_id in word_ids if word_id is not None}

        if len(present_word_ids) < len(sentence_labels):
            truncated_count += 1

    return truncated_count


def preview_alignment(
    sentence_tokens: List[str],
    sentence_labels: List[str],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
):
    """
    Print a readable token-label alignment for one sentence.
    Useful for debugging.
    """
    encoding = tokenize_and_align_labels(
        [sentence_tokens],
        [sentence_labels],
        tokenizer,
        label2id,
        padding=False,
        truncation=True,
    )

    input_ids = encoding["input_ids"][0]
    aligned_label_ids = encoding["labels"][0]

    mbert_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("Original tokens:")
    print(sentence_tokens)
    print()

    print("Original labels:")
    print(sentence_labels)
    print()

    print("mBERT token-label alignment:")
    for token, label_id in zip(mbert_tokens, aligned_label_ids):
        if label_id == IGNORE_INDEX:
            label = "-100"
        else:
            label = id2label[label_id]

        print(f"{token:15s} {label}")
