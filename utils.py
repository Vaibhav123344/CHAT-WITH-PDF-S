import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_files: list[bytes]) -> str:
    text = ""
    for data in pdf_files:
        reader = PdfReader(data)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vector_store(chunks: list[str], index_path: str = "faiss_index") -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    store = FAISS.from_texts(chunks, embeddings=embeddings, allow_dangerous_deserialization=True)
    store.save_local(index_path)

def load_vector_store(index_path: str = "faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    return FAISS.load_local(index_path, embeddings)

def make_qa_chain() -> any:
    template = """
    Answer the question using only the provided context. 
    If the answer is not in context, respond: "Answer not available."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)