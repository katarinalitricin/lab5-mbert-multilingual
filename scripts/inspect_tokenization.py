from transformers import AutoTokenizer

MODEL_CHECKPOINT = "bert-base-multilingual-cased"

sentence = "Pouvez-vous donner les mêmes garanties au sein de l’Union Européene"

ud_tokens = [
    "Pouvez", "-", "vous", "donner", "les", "mêmes", "garanties",
    "au", "sein", "de", "l", "’", "Union", "Européene"
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("Raw sentence:")
    print(sentence)
    print()

    print("UD-style tokenization:")
    print(ud_tokens)
    print()

    print("mBERT tokenization from raw string:")
    print(tokenizer.tokenize(sentence))
    print()

    print("mBERT tokenization from UD tokens:")
    encoding = tokenizer(
        ud_tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    print(tokens)
    print()

    print("Offsets:")
    for token, offset in zip(tokens, encoding["offset_mapping"]):
        print(f"{token:15s} {offset}")


if __name__ == "__main__":
    main()