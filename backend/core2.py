from dotenv import load_dotenv
load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as pineconeLangchian
from langchain import hub
from const import INDEX_NAME, MODEL


def run_llm(query):
    embeddings = OpenAIEmbeddings(model=MODEL)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # connect to Pinecone index already set up  without needing to recreate the vector store from scratch
    vectorstore = pineconeLangchian.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

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
    
    

