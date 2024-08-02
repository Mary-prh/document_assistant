from dotenv import load_dotenv
load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub


def run_llm(query):
    INDEX_NAME = 'langchain-doc-index'
    MODEL = 'text-embedding-3-small'
    embeddings = OpenAIEmbeddings(model=MODEL)
    chat = ChatOpenAI(verbose=True, temperature=0)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retreival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") # to download augmentation prompts

    combine_docs_chain = create_stuff_documents_chain(llm=chat , prompt=retreival_qa_chat_prompt)# Create a chain for passing a list of Documents to a model
    
    retreival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)
    result = retreival_chain.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    query = input("Proceed asking questions:")
    result = run_llm(query=query)
    print(result['answer'])
    
    

