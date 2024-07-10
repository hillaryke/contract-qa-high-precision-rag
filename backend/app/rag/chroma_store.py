import os
import shutil
import logging
import coloredlogs

from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma


class ChromaStore:
  def __init__(self, chroma_path="./chroma", data_path=None, embeddings=None):
    self.chroma_path = chroma_path
    self.data_path = data_path
    self.embeddings = embeddings if embeddings else OpenAIEmbeddings()
    self.logger = logging.getLogger(__name__)
    coloredlogs.install(level="WARNING", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

  def save_to_chroma(self, docs):
    try:
      self.logger.info("Clearing out the chroma database.")
      if os.path.exists(self.chroma_path):
        shutil.rmtree(self.chroma_path)
        
        # Add a delay to ensure the directory is fully removed
        import time
        time.sleep(1)
      
      self.logger.info("Creating a new chroma database.")
      vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings,
                        persist_directory=self.chroma_path)
      return vectorstore
    except Exception as e:
      self.logger.error(f"Failed to save to chroma: {e}")
      return None

  def initialize_vectorstore(self, chunks):
    try:
      vectorstore = self.save_to_chroma(chunks)
      return vectorstore
    except Exception as e:
      self.logger.error(f"Failed to initialize vectorstore: {e}")
      return None

  def get_vectorstore(self):
    try:
      self.logger.info("Loading the vectorstore from chroma db.")
      vectorstore = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
      return vectorstore
    except Exception as e:
      self.logger.error(f"Failed to get vectorstore: {e}")
      return None

  def get_retriever(self, vectorstore: Chroma = None, similarity_threshold: float = 0.8, similarity_count: int = 5, sources: bool = False):
    try:
      if vectorstore is None:
        self.logger.error("Vectorstore is not provided.")
        return None
      retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                         search_kwargs={'score_threshold': similarity_threshold,
                                "k": similarity_count})
      return retriever
    except Exception as e:
      self.logger.error(f"Failed to get retriever: {e}")
      return None

  def load_documents_from_dir(self, glob: str = "*.docx"):
    try:
      self.logger.info(f"Loading documents from {self.data_path}")
      loader = DirectoryLoader(self.data_path, glob)
      documents = loader.load()
      self.logger.info(f"Loaded {len(documents)} documents")
      return documents
    except Exception as e:
      self.logger.error(f"Failed to load documents from directory: {e}")
      return []