import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import FakeEmbeddings
from pydantic import SecretStr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
## load the OpenRouter api key
openrouter_api_key=os.getenv("OPENROUTER_API_KEY","")
llm=ChatOpenAI(
    model="google/gemini-2.0-flash-exp:free",
    api_key=SecretStr(openrouter_api_key),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}

    """
)
def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading documents..."):
            st.session_state.loader=PyPDFDirectoryLoader("./pdfs") # data ingestion step
            st.session_state.docs=st.session_state.loader.load() # complete document loading
            
            if not st.session_state.docs:
                st.error("No PDF files found in ./pdfs directory!")
                return
            
            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            
            if not st.session_state.final_documents:
                st.error("No documents to process!")
                return
            
            st.session_state.embeddings=FakeEmbeddings(size=384)
            st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter yoour query from the documents:")
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")

import time
if user_prompt and "vectors" in st.session_state:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"response time : {time.process_time()-start}")
    st.write(response['answer'])
    ## with a streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")
elif user_prompt:
    st.warning("Please click 'Document Embedding' button first to load the documents.")