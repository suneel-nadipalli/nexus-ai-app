# Nexus AI - Choose Your Own Adventure App

By Suneel Nadipalli

Deployed URL: https://nexus-ai-app-ad5ec9c48e49.herokuapp.com/

## Project Overview
This project involves building a Web App to facilitate an intelligent AI-Based Choose Your Own Adventure App. Users can select a story from the available library, cycle through pages, select any generated options and tread a new story path.    

## Data Preparation

1. **PDF Document Compilation**
   - Collect story PDFs from across the web

2. **Generate and store embeddings and pages in MongoDB**
   - Use the OpenAI() embedding function to store the embeddings as well as the pages in MongoDB. 

## Model Training

### Summarization

- **Dataset**: Composed of [Text, Summary]
- **Models**:
  1. T5 Model - From Scratch
  2. Falcon 1B - Prertained
  3. RAG

### Classification

- **Dataset**: Composed of [Text, Decision]
- **Models**:
  1. T5 Model - From Scratch
  2. Mistral - Fine-Tuned LLM
  3. RAG

### Options Generation

- **Dataset**: Composed of [Text, Paths]
- **Models**:
  1. Mistral - Fine-Tuned LLM
  2. RAG

### Path/Text Generation

- **Dataset**: Composed of [Text, Decision]
- **Models**:
  1. LSTM - From Scratch
  2. Mistral - Unstructured LLM
  3. RAG

## Application Architecture

- **Frontend**: Built with HTML, CSS, JS.
- **Backend**: A Flask application deployed on Heroku to manage data processing and model interactions.
 
## Conclusion

This web app was able to facilitate the basic functioning of a Choose Your Own Adventure book by way of summarzing books into plot points, generating options for pivotal story moments and creating new story paths.  
