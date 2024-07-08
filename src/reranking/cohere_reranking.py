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
    Re-ranks documents retrieved from a vector store using Cohere's language model for improved relevance.

    This function first retrieves the top `k` documents from the vector store based on similarity to the query.
    It then uses Cohere's re-ranking model to re-rank these documents, aiming to improve the relevance of the
    retrieved documents to the query. The re-ranking is performed by a ContextualCompressionRetriever, which
    combines the initial retrieval with Cohere's re-ranking capabilities.

    Parameters:
        vectorstore (VectorStore): The vector store from which documents are retrieved. This object must
                                   implement an `as_retriever` method that returns a retriever object.
        query (str): The query string used to retrieve and re-rank documents.
        k (int, optional): The number of top documents to retrieve and re-rank. Defaults to 5.

    Returns:
        ContextualCompressionRetriever: An object capable of retrieving and re-ranking documents based on
                                        the query, using Cohere's re-ranking model for improved relevance.
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
    return compression_retriever