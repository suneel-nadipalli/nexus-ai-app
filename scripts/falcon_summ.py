
from transformers import pipeline

import PyPDF2

def read_pages(pdf_file):
    pages = []
    
    reader = PyPDF2.PdfReader(pdf_file)
    
    for page_number in range(len(reader.pages)):
    
        page = reader.pages[page_number]
    
        page_content = page.extract_text()

        pages.append(page_content)

    return pages

def prep_pipeline():
     summarizer = pipeline("summarization", model=f"Falconsai/text_summarization")

     return summarizer

def gen_summary(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

    return summary