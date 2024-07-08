import os
import sys

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag.chroma_store import ChromaStore
from rag.utils import pretty_print_docs
from rag.chunking_strategies import chunk_by_recursive_split
from misc import Settings

load_dotenv()

GENERATOR_TEMPLATE = Settings.GENERATOR_TEMPLATE

def initialize_rag_pipeline():
    # Load the documents from the data directory.
    # documents = load_documents_from_dir("data/content")
    chroma_store = ChromaStore(data_path="data/content")
    documents = chroma_store.load_documents_from_dir()

    chunks = chunk_by_recursive_split(documents, chunk_size=400)

    vectorstore = chroma_store.initialize_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        }
    )
    # compressor = CohereRerank()
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever
    # )

    # compressed_docs = compression_retriever.invoke(
    #     question
    # )
    # pretty_print_docs(compressed_docs)


    PROMPT = PromptTemplate(
        template=GENERATOR_TEMPLATE, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT, "verbose": False}

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa
    
def query_rag_pipeline(qa, query):
    response = qa.invoke({"query": query})
    answer = response["result"]
    return answer
    
# Example usage
if __name__ == "__main__":
    qa = initialize_rag_pipeline()
    query = "Can the Advisor charge for meal time?"
    answer = query_rag_pipeline(qa, query)
    print("Answer:", answer)