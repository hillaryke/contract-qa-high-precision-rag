import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory of `backend` to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.rag.chroma_store import ChromaStore
from src.env_loader import load_env
load_env()

class TestChromaStore(unittest.TestCase):
  @patch('backend.app.rag.chroma_store.os.path.exists')
  @patch('backend.app.rag.chroma_store.shutil.rmtree')
  @patch('backend.app.rag.chroma_store.Chroma.from_documents')
  def test_save_to_chroma_success(self, mock_from_documents, mock_rmtree, mock_exists):
    # Setup
    mock_exists.return_value = True
    chroma_store = ChromaStore()
    docs = ["doc1", "doc2"]

    # Execute
    result = chroma_store.save_to_chroma(docs)

    # Verify
    mock_exists.assert_called_once_with(chroma_store.chroma_path)
    mock_rmtree.assert_called_once_with(chroma_store.chroma_path)
    mock_from_documents.assert_called_once()
    self.assertIsNotNone(result)

  @patch('backend.app.rag.chroma_store.os.path.exists')
  @patch('backend.app.rag.chroma_store.shutil.rmtree')
  @patch('backend.app.rag.chroma_store.Chroma.from_documents', side_effect=Exception("Error"))
  def test_save_to_chroma_failure(self, mock_from_documents, mock_rmtree, mock_exists):
    # Setup
    mock_exists.return_value = True
    chroma_store = ChromaStore()
    docs = ["doc1", "doc2"]

    # Execute
    result = chroma_store.save_to_chroma(docs)

    # Verify
    self.assertIsNone(result)

if __name__ == '__main__':
  unittest.main()