from ssf_model import BertForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, EvalPrediction, \
        Trainer, TrainingArguments
import torch

def main():

    model_ckpt = 'bert-base-uncased'

    dataset = load_dataset('glue', 'mnli')
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {id: label for label, id in label2id.items()}

    sentence1_key, sentence2_key = ("premise", "hypothesis")

    dataset = dataset.map(
        lambda examples: tokenizer(*((examples["premise"], 
            examples["hypothesis"])), padding=False,
            truncation=True),
        batched=True)


    config = AutoConfig.from_pretrained(
        model_ckpt,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
        # finetuning_task=data_args.dataset_name,
        # revision=model_args.model_revision,
    )

    device = torch.device("cpu")


    model = BertForSequenceClassification.from_pretrained(
        model_ckpt,
        config=config,
        # revision=model_args.model_revision,
    ).to(device)

    batch_size = 16
    logging_steps = len(dataset["train"]) // batch_size

    training_args = TrainingArguments(
            output_dir="training_save_data",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.1,
            disable_tqdm=False,
            logging_steps=logging_steps,
            push_to_hub=False,
            log_level="error",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_f1",
            #load_best_model_at_end=args.early_stopping
    )

    trainer = Trainer(
            model=model, 
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation_matched"],
            tokenizer=tokenizer)

    trainer.train()


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


if __name__ == "__main__":
    main()
