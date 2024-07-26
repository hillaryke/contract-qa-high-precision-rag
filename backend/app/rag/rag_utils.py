import logging
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os
from misc import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# Retrieve
question = "Is escrow amount greater than the Retention Amount?"

def get_unique_union(documents: list[list]):
    """
    Creates a unique union of retrieved documents.
    
    Parameters:
        documents (list[list]): A list of lists containing document objects.
    
    Returns:
        list: A list of unique document objects.
    """
    try:
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]
    except Exception as e:
        logger.error(f"Failed to get unique union of documents: {e}")
        return []

def get_context_by_multiple_queries(question, llm=None, retriever=None):
    """
    Generates multiple queries from a single question and retrieves documents for each query.
    
    Parameters:
        question (str): The original user question.
        llm (ChatOpenAI, optional): The language model to use for generating queries.
        retriever (callable, optional): The retriever to use for fetching documents.
    
    Returns:
        list: A list of retrieved document objects.
    """
    try:
        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        retrieval_chain = generate_queries | retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question": question})
        return docs
    except Exception as e:
        logger.error(f"Failed to get context by multiple queries: {e}")
        return []

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to multiple lists of ranked documents.
    
    Parameters:
        results (list[list]): A list of lists containing ranked document objects.
        k (int, optional): A parameter used in the RRF formula. Defaults to 60.
    
    Returns:
        list: A list of tuples, each containing a document object and its fused score.
    """
    try:
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=itemgetter(1), reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results
    except Exception as e:
        logger.error(f"Failed to apply reciprocal rank fusion: {e}")
        return []

def create_rank_fusion_chain(question, llm=None, retriever=None):
    """
    Creates a retrieval chain that applies rank fusion to the results of multiple queries.
    
    Parameters:
        question (str): The original user question.
        llm (ChatOpenAI, optional): The language model to use for generating queries.
        retriever (callable, optional): The retriever to use for fetching documents.
    
    Returns:
        callable: A retrieval chain that applies rank fusion.
    """
    try:
        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
        return retrieval_chain_rag_fusion
    except Exception as e:
        logger.error(f"Failed to create rank fusion chain: {e}")
        return lambda x: []

def generate_answer(question, context, llm=None):
    """
    Generates an answer to a question based on the provided context.
    
    Parameters:
        question (str): The user question.
        context (str): The context to use for generating the answer.
        llm (ChatOpenAI, optional): The language model to use for generating the answer.
    
    Returns:
        str: The generated answer.
    """
    try:
        # RAG
        template = Settings.GENERATOR_TEMPLATE

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            {"context": itemgetter("context"), "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = final_rag_chain.invoke({"context": context, "question": question})
        return answer
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        return "An error occurred while generating the answer."