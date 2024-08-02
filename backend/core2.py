from dotenv import load_dotenv
load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as pineconeLangchian
from langchain import hub
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from backend.const import INDEX_NAME, MODEL


def run_llm(query, chat_history):
    embeddings = OpenAIEmbeddings(model=MODEL)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # connect to Pinecone index already set up  without needing to recreate the vector store from scratch
    vectorstore = pineconeLangchian.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retreival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") # to download augmentation prompts

    combine_docs_chain = create_stuff_documents_chain(llm=chat , prompt=retreival_qa_chat_prompt)# Create a chain for passing a list of Documents to a model
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    retreival_chain = create_retrieval_chain(retriever=history_aware_retriever,
                                             combine_docs_chain=combine_docs_chain)
    result = retreival_chain.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result

if __name__ == "__main__":
    query = input("Proceed asking questions:")
    result = run_llm(query=query)
    print(result['result'])
    
    

