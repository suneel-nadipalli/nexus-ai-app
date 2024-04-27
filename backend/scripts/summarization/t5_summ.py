from datasets import load_dataset

from transformers import AutoTokenizer

from transformers import DataCollatorForSeq2Seq

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from transformers import pipeline

checkpoint = "Falconsai/text_summarization"

output_dir = "falcon-summ"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

import numpy as np

import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def preprocess_function(examples, max_length=1024, max_target_length=128):

    prefix = "summarize: "

    inputs = [prefix + doc for doc in examples["text"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def prep_data():
    billsum = load_dataset("billsum", split="ca_test")

    billsum = billsum.train_test_split(test_size=0.2)

    return billsum

def prep_model():

    billsum = prep_data()

    tokenized_billsum = billsum.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=30,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer

def train_model(trainer):
    
        trainer.train()
    
        trainer.save_model(output_dir)
    
        trainer.push_to_hub()

def prep_pipeline():
     summarizer = pipeline("summarization", model=f"suneeln-duke/{output_dir}")

     return summarizer

def gen_summary(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

    return summary