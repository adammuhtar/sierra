# Standard library imports
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import uuid

# Third party imports
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Local application imports
from src.vectordb.vectordb import VectorDB, VectorDBEntry, VectorDBGenerator


class TestVectorDB(unittest.TestCase):
    def test_vectordb_entry(self):
        documents = [
            {
                "source": "doc1",
                "contents": [
                    Document(
                        page_content="test content",
                        metadata={
                            "page": 1,
                            "embedding": "embedding",
                            "doc_uuid": uuid.uuid4(),
                        },
                    )
                ],
            }
        ]
        entry = VectorDBEntry(corpus_name="Test Corpus", documents=documents)
        self.assertEqual(entry.corpus_name, "Test Corpus")
        self.assertEqual(entry.documents, documents)

    def test_vectordb_result(self):
        documents = [
            {
                "source": "doc1",
                "contents": [
                    Document(
                        page_content="test content",
                        metadata={
                            "page": 1,
                            "embedding": "embedding",
                            "doc_uuid": uuid.uuid4(),
                        },
                    )
                ],
            }
        ]
        entry = VectorDBEntry(corpus_name="Test Corpus", documents=documents)
        vectordb_result = VectorDB(entries=[entry])

        expected_dict = [{"Test Corpus": documents}]
        self.assertEqual(vectordb_result.to_dict(), expected_dict)

        file_path = "test_vectordb.json"
        vectordb_result.save_to_json(file_path)
        loaded_vectordb_result = VectorDB.load_from_json(file_path)
        self.assertEqual(
            vectordb_result.entries[0].corpus_name,
            loaded_vectordb_result.entries[0].corpus_name,
        )


# class TestVectorDBGenerator(unittest.TestCase):
#     @patch("vectordb.vectordb.pymupdf.open")
#     @patch("vectordb.vectordb.SentenceTransformer")
#     def test_generate_vectordb(self, mock_encoder, mock_open):
#         mock_encoder.encode = Mock(return_value="embedding")
#         encoder = mock_encoder()
#         vdb = VectorDBGenerator(encoder=encoder, device="cpu", resolution="page")

#         pdf_files = [{"company1": [Path("file1.pdf"), Path("file2.pdf")]}]
#         mock_open.return_value = Mock()

#         result = vdb.generate_vectordb(files=pdf_files)
#         self.assertIsInstance(result, VectorDB)
#         self.assertEqual(len(result.entries), 1)
#         self.assertEqual(result.entries[0].corpus_name, "company1")

#     @patch("vectordb.vectordb.pymupdf.open")
#     @patch("vectordb.vectordb.SentenceTransformer")
#     def test_generate_vectordb_entry(self, mock_encoder, mock_open):
#         mock_encoder.encode = Mock(return_value="embedding")
#         encoder = mock_encoder()
#         vdb = VectorDBGenerator(encoder=encoder, device="cpu", resolution="block")

#         mock_doc = Mock()
#         mock_doc.get_text.return_value = "This is a test."
#         mock_open.return_value = mock_doc

#         pdf_files = [Path("file1.pdf")]
#         documents = vdb._generate_vectordb_from_file(files=pdf_files)
#         self.assertEqual(len(documents), 1)
#         self.assertIn("source", documents[0])
#         self.assertIn("contents", documents[0])


if __name__ == "__main__":
    unittest.main()
