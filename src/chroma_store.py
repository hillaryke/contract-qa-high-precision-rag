from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def save_to_chroma(docs: list[Document]):
    # Clear out the database first.
    logger.info("Clearing out the chroma database.")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    logger.info("Creating a new chroma database.")
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    return vectorstore

# TODO - modify to allow any document loaded
def initialize_vectorstore():
    # Load the documents from the data directory.
    documents = load_documents_from_dir("data/content")
    # Split the documents into chunks.
    chunks = split_text(documents)
    # Save the chunks to the chroma store.
    vectorstore = save_to_chroma(chunks)
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
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

def load_documents_from_dir(DATA_PATH: str, glob: str = "*.docx"):
  logger.info(f"Loading documents from {DATA_PATH}")
  loader = DirectoryLoader(DATA_PATH, glob)
  documents = loader.load()
  return documents