from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
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
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def retrieve_context(query, retriever):
    """Retrieves relevant context for a given query from your knowledge base or documents."""
    docs = retriever.invoke(query)
    return docs

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