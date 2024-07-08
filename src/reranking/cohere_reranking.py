from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain_cohere import ChatCohere
from src.utils import pretty_print_docs

llm = ChatCohere(
    model="command",
    temperature=0,
)

def cohere_rerank(vectorstore, query, k=5):
    """
    Performs re-ranking of documents retrieved from a vector store using Cohere's language model.
    
    This function retrieves the top `k` documents from the vector store based on similarity to the query.
    It then uses Cohere's language model to re-rank these documents, aiming to improve the relevance of the
    retrieved documents to the query.
    
    Parameters:
        vectorstore (VectorStore): The vector store from which documents are retrieved.
        query (str): The query string used to retrieve and re-rank documents.
        k (int, optional): The number of top documents to retrieve and re-rank. Defaults to 5.
    
    Returns:
        None: The function currently does not return any value. It is intended to demonstrate the process
        of re-ranking and may be modified to return the re-ranked documents.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        }
    )
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )