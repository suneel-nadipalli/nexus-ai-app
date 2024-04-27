import os
import json
import re

import openai

import langchain

import langchain.document_loaders

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# read from config.ini file


import PyPDF2

def read_pages(pdf_file):
    pages = []
    
    reader = PyPDF2.PdfReader(pdf_file)
    
    for page_number in range(len(reader.pages)):
    
        page = reader.pages[page_number]
    
        page_content = page.extract_text()

        pages.append(page_content)

    return pages

def get_chunks(file_path):
    
    loader = PyPDFLoader(file_path)
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def get_vectordb(chunks, CHROMA_PATH):

    CHROMA_PATH = f"../../data/chroma/{CHROMA_PATH}"

    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    else:
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
        )

        db.persist()

        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
    return db

def classify_dec(text, db):

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    query_text = f"""

    Classify whether the given chunk involves a decision that will effect the story or not.

    A decision is defined as when the character goes about making a choice between two or more options. 
    The decision should be significant enough to affect the story in a major way.
    It doesn't really involve emotions, feelings or thoughts, but what the character does, or what happens to them.
    This involes interactions between characters, or the character and the environment.
    What isn't a decision is chunks describing the setting, or the character's thoughts or feelings.

    Return the answer as the corresponding decision label "yes" or "no"

    {text}

    """

    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    return (response_text)
