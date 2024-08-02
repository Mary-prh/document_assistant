from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os


embedding = OpenAIEmbeddings(model="text-embedding-3-small") # the vectorDb has also the same model for embed the document


def ingest_docs():
    print(f'Current working directory: {os.getcwd()}')
    path = os.path.join(os.getcwd(), 'langchain-docs', 'api.python.langchain.com', 'en', 'latest')
    print(f'Constructed path: {path}')
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
    
    loader = ReadTheDocsLoader(path, encoding = 'UTF-8')
    raw_text = loader.load()
    print(f'length of data is: {len(raw_text)}')
    # ensuring that the chunks meaningful and coherent by Recursive Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50) 
    document = text_splitter.split_documents(raw_text)
    for doc in document:
        # Metadata includes additional information about the document, such as its source URL
        new_url = doc.metadata['source']
        # transform the local file path into a web URL
        new_url = new_url.replace('langchain-docs', 'https:/')
        doc.metadata.update({'source': new_url})
    print(f'Going to add {len(document)} to Pinecone')
    PineconeVectorStore.from_documents(embedding= embedding, documents= document,
                                       index_name= 'langchain-doc-index')

if __name__ == '__main__':
    ingest_docs()
