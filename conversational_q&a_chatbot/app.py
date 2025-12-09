# by the way , i am GORIB , so i use opensource model to implement this , using openrouter api key to access free models
import streamlit as st
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Simple embeddings class
class SimpleEmbeddings(Embeddings):
    def __init__(self, size=384):
        self.size = size
    
    def embed_documents(self, texts):
        return [np.random.rand(self.size).tolist() for _ in texts]
    
    def embed_query(self, text):
        return np.random.rand(self.size).tolist()

embeddings = SimpleEmbeddings(size=384)

## Set up Streamlit
st.title("Conversational Q&A Chatbot with PDF")
st.write("Upload a PDF and have a conversation about its content")

## Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY", "")

if not api_key:
    st.error("Please set OPENROUTER_API_KEY in your .env file")
# Initialize LLM
llm = ChatOpenAI(
    model="tngtech/tng-r1t-chimera:free",
    api_key=SecretStr(api_key),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)



# Session ID
session_id = st.text_input("Session ID", value="default_session")

# Manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file:", type='pdf')

# Process uploaded file
if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save temporary PDF
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        
        # Load and split document
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\\n\\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Create RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Function to get session history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        # Create conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        st.success("PDF processed successfully! You can now ask questions.")
        
        # Display chat history
        if session_id in st.session_state.store:
            st.subheader("Chat History")
            for message in st.session_state.store[session_id].messages:
                if message.type == "human":
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)
        
        # User input
        user_input = st.chat_input("Ask a question about the PDF:")
        
        if user_input:
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Get response
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
            
            # Display assistant response
            st.chat_message("assistant").write(response['answer'])
            
            # Rerun to update chat history display
            st.rerun()
else:
    st.info("Please upload a PDF file to start chatting.")
# Adding a temp.pdf for checking , this is mini project of my collage in Operating System Lab
