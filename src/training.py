from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src.metrics import compute_accuracy_metrics


MODEL_CHECKPOINT = "bert-base-multilingual-cased"


def build_model(label_list, id2label, label2id, model_checkpoint=MODEL_CHECKPOINT):
    """
    Load mBERT with a token classification head.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    return model


def build_trainer(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
):
    """
    Build a HuggingFace Trainer for PoS tagging.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_accuracy_metrics,
    )

    return trainer