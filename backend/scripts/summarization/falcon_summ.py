
from transformers import pipeline

def prep_pipeline():
     summarizer = pipeline("summarization", model=f"Falconsai/text_summarization")

     return summarizer

def gen_summary(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

    return summary