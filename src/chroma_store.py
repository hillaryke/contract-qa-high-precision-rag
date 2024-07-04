from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

import os
import shutil

import logging
import coloredlogs

logger = logging.getLogger(__name__)

coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

CHROMA_PATH = "chroma"
DATA_PATH = "data/content"

def save_to_chroma(docs: list[Document], embeddings = OpenAIEmbeddings()):
    # Clear out the database first.
    logger.info("Clearing out the chroma database.")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    logger.info("Creating a new chroma database.")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings,
                                        persist_directory=CHROMA_PATH)
    return vectorstore


def initialize_vectorstore(chunks, embeddings = OpenAIEmbeddings()):
    # Save the chunks to the chroma store.
    vectorstore = save_to_chroma(chunks, embeddings)
    return vectorstore

def get_vectorstore():
    """Get the vector store from the chroma database."""
    logger.info("Loading the vectorstore from chroma db.")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    return vectorstore
def get_retriever(vectorstore: Chroma = None, similarity_threshold: float = 0.8, similarity_count: int = 5,  sources: bool = False):
    """Retrieves relevant context for a given query from the knowledge base or documents."""
    
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={'score_threshold': similarity_threshold,
                                                       "k": similarity_count})
    return retriever

def load_documents_from_dir(DATA_PATH: str, glob: str = "*.docx"):
  print(f"--INFO-- Loading documents from {DATA_PATH}")
  loader = DirectoryLoader(DATA_PATH, glob)
  documents = loader.load()
  print(f"--INFO-- Loaded {len(documents)} documents")
  return documents