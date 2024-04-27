import os

import numpy as np

import pandas as pd

import os

from tqdm import tqdm

from transformers import pipeline

from transformers import AutoTokenizer, FalconForCausalLM

import torch

from datasets import Dataset

from peft import LoraConfig

from trl import SFTTrainer

from transformers import ( 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          )
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

def generate_prompt(data_point):
    return f"""### Instruction:
            Classify whether the given chunk involves a decision that will effect the story or not.
            A decision is defined as when the character goes about making a choice between two or more options. 
            The decision should be significant enough to affect the story in a major way.
            It doesn't really involve emotions, feelings or thoughts, but what the character does, or what happens to them.
            This involes interactions between characters, or the character and the environment.
            What isn't a decision is chunks describing the setting, or the character's thoughts or feelings.
            Return the answer as the corresponding decision label "yes" or "no"
            
            ### Text:
            {data_point["text"]}
            
            ### Decision:
            {data_point["decision"]}
            """

def generate_test_prompt(data_point):
    return f"""### Instruction:
            Classify whether the given chunk involves a decision that will effect the story or not.
            A decision is defined as when the character goes about making a choice between two or more options. 
            The decision should be significant enough to affect the story in a major way.
            It doesn't really involve emotions, feelings or thoughts, but what the character does, or what happens to them.
            This involes interactions between characters, or the character and the environment.
            What isn't a decision is chunks describing the setting, or the character's thoughts or feelings.
            Return the answer as the corresponding decision label "yes" or "no"
            
            ### Text:
            {data_point["text"]}
            
            ### Decision:
            """

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens = 1, 
                        temperature = 0.0,
                       )
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]['generated_text'].split("=")[-1].lower()
        if "yes" in answer:
            y_pred.append("yes")
        elif "no" in answer:
            y_pred.append("no")
        else:
            y_pred.append("none")
    return y_pred

def evaluate(y_true, y_pred):
    labels = ['yes', 'no', 'none']
    mapping = {"yes": 1, "no": 0, 'none':2}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)


def prep_data():
    filename = '../../data/output/decisions.csv'

    df = pd.read_csv(filename, encoding="utf-8", encoding_errors="replace")

    df = df[['text', 'decision']]

    X_train = list()
    
    X_test = list()
    
    for decision in ["yes", "no"]:
        train, test  = train_test_split(df[df.decision==decision], 
                                        train_size=.8,
                                        test_size=.2, 
                                        random_state=42)
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    
    X_test = pd.concat(X_test)

    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    
    X_eval = df[df.index.isin(eval_idx)]
    
    X_eval = (X_eval
            .groupby('decision', group_keys=False)
            .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    
    X_train = X_train.reset_index(drop=True)

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), 
                       columns=["text"])
    
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), 
                        columns=["text"])

    y_true = X_test.decision
    
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    train_data = Dataset.from_pandas(X_train)
    
    eval_data = Dataset.from_pandas(X_eval)

    return train_data, eval_data


def prep_model():
    model_name = "Rocketknight1/falcon-rw-1b"

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = FalconForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config, 
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            padding_side="left",
                                            add_bos_token=True,
                                            add_eos_token=True,
                                            )

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prep_trainer():
    OUTPUT_DIR = "falcon-clf"

    train_data, eval_data = prep_data()

    model, tokenizer = prep_model()

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=1024,
    )

    return trainer

def train_model():

    trainer = prep_trainer()

    trainer.train()

    trainer.model.save_pretrained("falcon-clf")

    trainer.push_to_hub()

def get_classifier():
    classifier = pipeline(model=f"suneeln-duke/falcon-clf", device_map="auto")

    return classifier

def classify_dec(text, classifier):

    text = generate_test_prompt({
        'text': text
    })
    
    result = classifier(text, pad_token_id=classifier.tokenizer.eos_token_id)
    
    answer = result[0]['generated_text'].split("=")[-1].lower()

    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"