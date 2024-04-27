from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

import os, sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append("..")

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

def prep_config(vs):

    retriever = vs.as_retriever(
                search_type = "similarity",
                search_kwargs = {"k": 3}
                )
    
    template = """Answer the question: {question} based only on the following context:
    context: {context}
    """

    output_parser = JsonOutputParser()

    prompt = PromptTemplate.from_template(template = template,
                        input_varaibles = ["context", "question"],
                        output_variables = ["answer"],)
    
    output_parser = StrOutputParser()

    
    
    model = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), 
                model_name = 'gpt-4',
                temperature=0.3)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retrieval_chain = (
        {"context": retriever | format_docs,  "question": RunnablePassthrough()}
        | prompt 
        | model 
        | output_parser
    )

    return retrieval_chain, output_parser


def gen_options(vs, text):

    retrieval_chain, output_parser = prep_config(vs)

    query = f"""
    Act as the author of a Choose Your Own Adventure Book. This book is special as it is based on existing material.
    Now, as with any choose your own adventure book, you'll have to generate decision paths based on the given story excerpt
    Your job is to generate 4 decision paths for the given point in the story.
    One among the 4 decision paths should be the original path, the other 3 should deviate from the original path in a sensible manner.
    The decision paths should be generated in a way that they are coherent with the existing story.
    Limit each decision path to a succint sentence.
    Return the 4 decision paths as a list of strings.

    Story Excerpt: {text}

    """

    response = retrieval_chain.invoke(query)

    return response

def gen_path(vs, text, decision):

    retrieval_chain, output_parser = prep_config(vs)

    query = f"""
    Act as the author of a Choose Your Own Adventure Book. This book is special as it is based on existing material.
    Now, as with any choose your own adventure book, you'll have to generate new story paths based on a relevant excerpt of the story and the decision taken.
    Your job is to generate the next part of the story based on the given part of the story and the decision taken.
    The new story path should be coherent with the existing story, and should be around 6-8 sentences.
    If the decision string is empty, your task is just to generate the next part of the story based on the given part of the story.
    Return the new story path as a string.

    Story Excerpt: {text}

    Decision: {decision}
    """

    response = retrieval_chain.invoke(query)

    return output_parser.parse(response)

def clf_seq(vs, text):

    retrieval_chain, output_parser = prep_config(vs)

    query = f"""
    Classify whether the given chunk involves a decision that will effect the story or not.

    A decision is defined as when the character goes about making a choice between two or more options. 
    The decision should be significant enough to affect the story in a major way.
    It doesn't really involve emotions, feelings or thoughts, but what the character does, or what happens to them.
    This involes interactions between characters, or the character and the environment.
    What isn't a decision is chunks describing the setting, or the character's thoughts or feelings.

    Return the answer as the corresponding decision label "yes" or "no"

    {text}
     """

    response = retrieval_chain.invoke(query)

    return output_parser.parse(response)

def summ(vs, text):

    retrieval_chain, output_parser = prep_config(vs)

    query = f"""
    Summarize the given text in a narrative manner as a part of storytelling.
    The summary should be around 3-4 sentences and should be coherent with the existing story.

    Return the summary as a string.
    {text}
     """

    response = retrieval_chain.invoke(query)

    return output_parser.parse(response)