"""
Implement langchain code to ingest our text into the Pinecone vector store
It 
"""
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


if __name__ == "__main__":
    pdf_path = './db/infant4.pdf'
    loader = PyPDFLoader(pdf_path)
    raw_text = loader.load()

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50) 
    document = text_splitter.split_documents(raw_text)
    
    print(f'Going to add {len(document)} to Pinecone')
    embedding = OpenAIEmbeddings(model="text-embedding-3-small") # the vectorDb has also the same model for embed the document

    PineconeVectorStore.from_documents(embedding= embedding, documents= document,
                                       index_name= os.environ["INDEX_NAME_PDF"])

    print("Finish!")