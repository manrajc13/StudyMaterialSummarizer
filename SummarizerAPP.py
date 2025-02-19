## Study material summarizer and QNA 

## Data ingestion --> Step 1

## pdf parsing 
from langchain_community.document_loaders import PyPDFLoader 

def load_document_fromPDF(file_path):  
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents 

## from youtube video

from langchain_community.document_loaders import YoutubeLoader



def load_document_fromYT(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info = False)
    documents = loader.load()
    return documents

## from website
from langchain_community.document_loaders import WebBaseLoader


def load_document_fromWebSite(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

## setting up the vector database 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_vectorStore(docs):  # creating embeddings and storing them in a vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(docs)
    vectorStore = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory="chroma_db")
    return vectorStore


## creating cheatSheet 

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser

System_prompt = ''' 
You are an expert at distilling complex information into concise, structured, and easy-to-understand cheat sheets. Your task is to analyze the provided text content and create a well-organized cheat sheet that includes the following:

1. **Overview**: Begin with a brief summary (5-6 sentences) of the content, highlighting its main purpose, scope, and key themes.

2. **Key Concepts**: 

3. **Fundamentals**:

4. **Key Definitions**:

5. **Examples**: If the content includes examples, include them under the relevant concepts or definitions to enhance understanding.

6. **Additional Notes**: Include any other important points, such as formulas, diagrams, or step-by-step processes, if applicable.

- Prioritize accuracy, clarity, and logical flow in your output.
'''
def generate_cheat_sheet(docs):
    content = " ".join([doc.page_content for doc in docs])
    # llm = OllamaLLM(model = 'llama2:latest')
    llm = ChatGroq(model = "llama-3.2-3b-preview", groq_api_key = api_key)
    prompt = ChatPromptTemplate([
        ("system", System_prompt),
        ("user", "Please create a cheat sheet for the following content: \n \n + {content} + \n\n"),
    ])
    output_parser = StrOutputParser()
    cheatsheet_chain = prompt|llm|output_parser

    response = cheatsheet_chain.invoke({"content":content})
    return response

## chatting with the content 

import os 
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

import streamlit as st 
working_dir = os.path.dirname(os.path.abspath(__file__))

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


def get_session_history(session_id : str)->BaseChatMessageHistory: # to get chat history
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_chain(vectorStore):
    llm =  ChatGroq(groq_api_key = api_key, model_name = "llama-3.1-8b-instant")
    retriever = vectorStore.as_retriever()
    contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood "
            "without the chat history. DO NOT answer the question, "
            "just reformulate it if needed and otherwise return it as it is"
        )
            
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
            "You are an assistant for question - answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If the context does not provide enough information the use your own domain knowledge to answer the question "
            "Explain your reasoning for the answer and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, 
            input_messages_key = "input",
            history_messages_key = "chat_history",
            output_messages_key = "answer"
        )
    
    return conversational_rag_chain

st.title("Study Material Summarizer with Cheat Sheet and QNA")

st.write("Upload your chapter PDF or url of the youtube video of the chapter to generate a cheat sheet")

session_id = st.text_input("Session ID: ", value = "default_session")

    ## Statefully manage chat history

if "store" not in st.session_state:
    st.session_state.store = {}

if "cheat_sheet_store" not in st.session_state:
    st.session_state.cheat_sheet_store = {}

upload_files_pdf = st.file_uploader("Choose a PDF file ", type = ["pdf"])

url = st.text_input("Enter YouTube URL")

web_url = st.text_input("Enter WebSite URL")

if (upload_files_pdf or url or web_url):
    document_uploaded = ""
    file_path = None
    if upload_files_pdf:
        document_uploaded = "PDF"
        file_path = f"{working_dir}/{upload_files_pdf.name}"
        with open(file_path, "wb") as f:
            f.write(upload_files_pdf.getbuffer())

    if (file_path is None and url):
        document_uploaded = "Youtube video"
        docs = load_document_fromYT(url)
    elif (file_path is not None):
        docs = load_document_fromPDF(file_path)
    
    elif (file_path is None and web_url):
        document_uploaded = "WebPage"
        docs = load_document_fromWebSite(web_url)
    
    if st.button("Generate Cheat Sheet"):
        cheat_sheet = generate_cheat_sheet(docs)
        st.session_state.cheat_sheet_store[session_id] = cheat_sheet  

    conversational_rag_chain = create_chain(setup_vectorStore(docs))
    session_history = get_session_history(session_id)

    if session_id in st.session_state.cheat_sheet_store:
        st.subheader("Generated Cheat Sheet:")
        st.write(st.session_state.cheat_sheet_store[session_id])

    print(st.session_state.store)

    for message in session_history.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"  # Determine the role
        with st.chat_message(role):  # Use 'user' or 'assistant'
            st.markdown(message.content)  # Correct way to access message content


    user_input = st.chat_input(f"Ask anything related to {document_uploaded}...")
    if user_input:
        response = conversational_rag_chain.invoke(
            {"input":user_input},
            config = {
                "configurable": {"session_id" : session_id}
            }, 
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            assistant_response = response['answer']
            st.markdown(assistant_response)
                
    
