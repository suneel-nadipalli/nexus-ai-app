from pymongo import MongoClient

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys, os, certifi

from dotenv import load_dotenv

from pathlib import Path

import PyPDF2

sys.path.append("..")

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")


def read_pages(pdf_file):
    pages = []
    
    reader = PyPDF2.PdfReader(pdf_file)
    
    for page_number in range(len(reader.pages)):
    
        page = reader.pages[page_number]
    
        page_content = page.extract_text()

        pages.append(page_content)

    return pages

def connect_to_mongo():
    ca = certifi.where()

    client = MongoClient(os.environ.get("MONGO_URI"), tlsCAFile=ca)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client

def insert_pages(pdf_file, client=None):

    pages = read_pages(pdf_file)

    name = Path(pdf_file).stem

    pages_dict = [{"text": page, "page": i, "source": name} for i, page in enumerate(pages)]

    if not client:
        client = connect_to_mongo()
    
    pages_db = client[os.environ.get("MONGO_PAGES_DB")]

    pages_collection = pages_db[f"{name}-pages"]

    pages_collection.insert_many(pages_dict)

    return list(pages_collection.find())


def get_pages(name, client=None):
    

    if not client:
        client = connect_to_mongo()
    
    pages_db = client[os.environ.get("MONGO_PAGES_DB")]

    if f"{name}-pages" not in pages_db.list_collection_names():
        print("inserting pages")
        return insert_pages(name, client=client)
    
    else:
        print("using existing page collection")
        pages_collection = pages_db[f"{name}-pages"]

        pages = list(pages_collection.find())

        return pages

def insert_vs(pdf_file, client=None):
    name = Path(pdf_file).stem

    if not client:
        client = connect_to_mongo()
    
    vs_db = client[os.environ.get("MONGO_VS_DB")]

    vs_collection = vs_db[f"{name}-vs"]

    loader = PyPDFLoader(pdf_file)
    
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                                   chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), 
                                disallowed_special=())

    # Create embeddings in atlas vector store
    vector_search = MongoDBAtlasVectorSearch.from_documents( 
                                    documents=chunks, 
                                    embedding= embeddings, 
                                    collection=vs_collection,
                                    index_name=os.environ.get("MONGO_INDEX_DB")
                                                        )
    
    return vector_search

def get_vs(name, client=None):
    
    if not client:
        client = connect_to_mongo()
    
    vs_db = client[os.environ.get("MONGO_VS_DB")]

    if f"{name}-vs" not in vs_db.list_collection_names():
        print("inserting vs")
        return insert_vs(name, client=client)
    
    else:
        print("using existing vs collection")
        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            os.environ.get("MONGO_URI"),
            os.environ.get("MONGO_VS_DB") + "." + f"{name}-vs",
            OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), 
                                    disallowed_special=()),
            index_name=os.environ.get("MONGO_INDEX_DB"),
        )

        return vector_search