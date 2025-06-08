import streamlit as st
from PyPDF2 import PdfReader # this helps to read pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

genai.configure( api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_docs( pdf_docs ):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter( chuck_size = 1000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings( model="models/embeddings-001") # embedding technique
    vector_store = FAISS.from_texts( text_chunks, embeddings = embeddings, allow_dangerous_deserialization=True)
    vector_store.save_local(" faiss index")

def get_conversetional_chains():
    promt_template = """
    asnwer the question in detail from provided context , make sure if answer not avaiable in provided pdfs then then say 'answer not available'
    don't provide context :\n {context}? \n
    Questions : {question}\n
    Answer: 
    
    """

    model = ChatGoogleGenerativeAI( model='gemini-pro', tempreture = 0.3)

    promt = PromptTemplate( template = promt_template, input_variables=["context" , "question"])
    chain = load_qa_chain( model , chain_type='stuff',promt=promt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001")

    new_db = FAISS.load_local("faiss index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversetional_chains()

    response = chain( {"input_documents" : docs, "question":user_question }, return_only_outputs=True  )
    print( response)
    st.write( "asnwer : ", response["output_text"])

# Streamlit App
def main():
    st.set_page_config("PDF Chat App")
    st.title("Chat with your PDFs using")

    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs") and pdf_docs:
        with st.spinner("Reading and processing..."):
            raw_text = get_pdf_docs(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs processed and indexed successfully!")

    question = st.text_input("Ask a question about the uploaded PDFs:")
    if question:
        with st.spinner("Searching for the answer..."):
            answer = user_input(question)
            st.markdown(f"Answer: {answer}")

if __name__ == "__main__":
    main()