import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory of `backend` to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming the pretty_print_docs function is in a module named utils
from backend.app.rag.utils import pretty_print_docs

class TestPrettyPrintDocs(unittest.TestCase):
  def setUp(self):
    # Mock document objects with a 'page_content' attribute
    self.doc1 = MagicMock()
    self.doc1.page_content = "Content of Document 1"
    self.doc2 = MagicMock()
    self.doc2.page_content = "Content of Document 2"
    self.docs = [self.doc1, self.doc2]

  def test_pretty_print_docs_with_valid_docs(self):
    # Test pretty_print_docs with a valid list of document objects
    with self.assertLogs() as captured:
      pretty_print_docs(self.docs)
      self.assertEqual(len(captured.records), 1)
      self.assertIn("Document 1:", captured.records[0].getMessage())
      self.assertIn("Document 2:", captured.records[0].getMessage())
      self.assertIn("Content of Document 1", captured.records[0].getMessage())
      self.assertIn("Content of Document 2", captured.records[0].getMessage())

  def test_pretty_print_docs_with_non_list(self):
    # Test pretty_print_docs with a non-list type
    with self.assertRaises(ValueError) as context:
      pretty_print_docs("not a list")
    self.assertTrue("The 'docs' parameter should be a list of document objects." in str(context.exception))

  def test_pretty_print_docs_with_missing_page_content(self):
    # Test pretty_print_docs with a document missing the 'page_content' attribute
    doc_without_page_content = MagicMock()
    with self.assertRaises(AttributeError) as context:
      pretty_print_docs([self.doc1, doc_without_page_content])
    self.assertTrue("Each document object must have a 'page_content' attribute." in str(context.exception))

if __name__ == '__main__':
  unittest.main()