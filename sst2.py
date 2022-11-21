from ssf_model import BertForSequenceClassification
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoConfig, EvalPrediction, \
        Trainer, TrainingArguments
import numpy as np
import torch

def main():

    def compute_metrics(p):
        preds, labels = p
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    model_ckpt = 'bert-base-uncased'

    # train_dataset = load_dataset('glue', 'mnli', split=['train[:64]'])[0]
    # val_dataset = load_dataset('glue', 'mnli',
    #         split=['validation_matched[:64]'])[0]

    # sentence1_key, sentence2_key = ("sentence", None)

    dataset = load_dataset('glue', 'sst2')
    label_list = dataset["train"].features["label"].names
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {id: label for label, id in label2id.items()}

    metric = load_metric('glue', 'sst2')

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # train_dataset = train_dataset.map(
    #     lambda examples: tokenizer(*((examples["premise"], 
    #         examples["hypothesis"])), padding=False,
    #         truncation=True),
    #     batched=True)

    # val_dataset = val_dataset.map(
    #     lambda examples: tokenizer(*((examples["premise"], 
    #         examples["hypothesis"])), padding=False,
    #         truncation=True),
    #     batched=True)

    dataset = dataset.map(
        lambda examples: tokenizer(examples["sentence"], truncation=True),
        batched = True
    )

    config = AutoConfig.from_pretrained(
        model_ckpt,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
        # finetuning_task=data_args.dataset_name,
        # revision=model_args.model_revision,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.2     
    device = torch.device("cuda")

    model = BertForSequenceClassification.from_pretrained(
        model_ckpt,
        config=config,
        tuning_mode='ssf',
        # revision=model_args.model_revision,
    ).to(device)

    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "ssf" in name or "pooler" in name:
            param.requires_grad = True

    pytorch_total_params = sum(p.numel() for p 
            in model.parameters() if p.requires_grad)

    print("Num Trainable Params: ", pytorch_total_params)

    batch_size = 16
    #logging_steps = len(dataset["train"]) // batch_size

    training_args = TrainingArguments(
            output_dir="training_save_data_2",
            overwrite_output_dir=True,
            num_train_epochs=40,
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.1,
            disable_tqdm=False,
    #        logging_steps=logging_steps,
            push_to_hub=False,
            log_level="error",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="accuracy",
            load_best_model_at_end=True
    )
    trainer = Trainer(
            model=model, 
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer)

    trainer.train()

    print(trainer.evaluate())


if __name__ == "__main__":
    main()
