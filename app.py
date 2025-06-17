import json
import os
import sys
import boto3
import streamlit as st

# Titan Embeddings and Text Model
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Store
from langchain.vectorstores import FAISS

# LangChain QA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Titan Embedding Model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

# Titan Text Model (for answering)
def get_titan_llm():
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={"maxTokenCount": 512, "temperature": 0.7},
    )
    return llm

# Ingest and split PDFs
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)

# Create FAISS vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA system using Titan model
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer['result']

# Streamlit UI
def main():
    st.set_page_config("Chat PDF")
    st.header("RAGify-powerful RAG applicaton ")

    user_question = st.text_input("Ask a question based on the PDF content:")

    with st.sidebar:
        st.title("Update or Create Vector Store")
        if st.button("Update Vectors"):
            with st.spinner("Getting the things done"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully!")

    if st.button("Generate"):
        with st.spinner("Getting response from Titan..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True  # ✅ Fix added here
            )
            llm = get_titan_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Done")

if __name__ == "__main__":
    main()
