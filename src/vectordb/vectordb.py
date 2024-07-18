# Standard library imports
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Union
import uuid

# Third party imports
from langchain.schema import Document
from PIL import Image
import pymupdf
from sentence_transformers import SentenceTransformer, util
import tqdm

# Local application imports 
from .extraction import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorDBEntry:
    """
    Class representing an entry in the vector database.

    Attributes:
        * corpus_name (`str`): The name of the corpus (e.g., firm name).
        * documents (`List[Dict[str, Union[str, List[Document]]]]`): List of
            documents associated with the corpus.
    """
    def __init__(
        self, corpus_name: str, documents: List[Dict[str, Union[str, List[Document]]]]
    ):
        """
        Initialise a VectorDBEntry object.

        Args:
            * corpus_name (`str`): The name of the corpus.
            * documents (`List[Dict[str, Union[str, List[Document]]]]`): List of
                documents associated with the corpus.

        Raises:
            ValueError: If corpus_name is not a string or documents is not a list.
        """
        if not isinstance(corpus_name, str):
            raise ValueError("corpus_name must be a string.")
        if not isinstance(documents, list):
            raise ValueError("documents must be a list.")
        
        self.corpus_name = corpus_name
        self.documents = documents


class VectorDB:
    """
    Class representing the result of a vector database generation.

    Attributes:
        * entries (`List[VectorDBEntry]`): List of entries in the vector database.
    """
    def __init__(self, entries: List[VectorDBEntry]):
        """
        Initialise a VectorDB object.

        Args:
            entries (`List[VectorDBEntry]`): List of entries in the vector database.

        Raises:
            `ValueError`: If entries is not a list of VectorDBEntry objects.
        """
        if not isinstance(entries, list) or not all(isinstance(entry, VectorDBEntry) for entry in entries):
            raise ValueError("entries must be a list of VectorDBEntry objects.")
        self.entries = entries
        
    def __repr__(self) -> str:
            return f"VectorDB(entries={self.entries})"
    
    def to_dict(self) -> List[Dict[str, Union[str, List[Document]]]]:
        """
        Convert the vector database result to a dictionary format.

        Returns:
            `List[Dict[str, Union[str, List[Document]]]]`: The vector database
                in dictionary format.
        """
        return [{entry.corpus_name: entry.documents} for entry in self.entries]

    def save_to_json(self, file_path: str) -> None:
        """
        Save the vector database result to a JSON file.

        Args:
            * file_path (`str`): Path to the file where the JSON data will be saved.

        Raises:
            `ValueError`: If file_path is not a string.
        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string.")
        
        data = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(data, f, default=self._json_serializer, indent=4)

    @classmethod
    def load_from_json(cls, file_path: str) -> "VectorDB":
        """
        Load a vector database result from a JSON file.

        Args:
            file_path (`str`): Path to the JSON file to be loaded.

        Returns:
            `VectorDB`: The loaded vector database result.

        Raises:
            `ValueError`: If file_path is not a string.
            `FileNotFoundError`: If the file does not exist.
        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string.")
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        entries = [
            VectorDBEntry(
                corpus_name=list(item.keys())[0], documents=list(item.values())[0]
            )
            for item in data
        ]
        return cls(entries=entries)

    @staticmethod
    def _json_serializer(obj):
        """
        Custom JSON serializer for objects not serializable by default.

        Args:
            `obj`: The object to serialize.

        Returns:
            `str`: The serialized object.

        Raises:
            `TypeError`: If the object type is not serializable.
        """
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Document):
            return obj.__dict__
        if isinstance(obj, uuid.UUID):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def semantic_search(
        self,
        query: str,
        encoder: SentenceTransformer,
        top_k: int = 32,
        min_results_length: int = 15
    ) -> List[Dict[str, Union[str, float, Document]]]:
        """
        Perform a semantic search on the vector database.

        Args:
            * query (`str`): The search query.
            * encoder (`SentenceTransformer`): Encoder used to generate embeddings.
            * top_k (`int`, optional): The number of top results to return.
                Defaults to 32.
            * min_results_length (`int`, optional): The minimum length of valid
                results in words. Defaults to 15.

        Returns:
            `List[Dict[str, Union[str, float, Document]]]`: The top search results
                with their scores and documents.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(min_results_length, int) or min_results_length <= 0:
            raise ValueError("min_results_length must be a positive integer.")

        query_embedding = encoder.encode(query)
        results = []

        for entry in self.entries:
            corpus_embeddings = [
                doc["metadata"]["embedding"] for doc in entry.documents["contents"]
            ]
            
            hits = util.semantic_search(
                query_embeddings=query_embedding,
                corpus_embeddings=corpus_embeddings,
                top_k=top_k,
                score_function=util.dot_score
            )[0]
            
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)

            for hit in hits:
                corpus_id = hit["corpus_id"]
                document = entry.documents["contents"][corpus_id]
                if len(document.page_content.split()) > min_results_length:
                    results.append(
                        {
                            "corpus_name": entry.corpus_name,
                            "score": hit["score"],
                            "document": document
                        }
                    )
        
        return results


class VectorDBGenerator:
    def __init__(
        self,
        encoder: SentenceTransformer,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
        resolution: Literal["block", "page"] = "block"
    ):
        """
        Initialise a VectorDB object with encoder model and device type.

        Args:
            * encoder (`SentenceTransformer`): Encoder for generating embeddings.
            * device (`Literal["cpu", "cuda", "mps"]`, optional): Device to run
                the encoder on. Defaults to "cpu".
            * resolution (`Literal["block", "page"]`, optional): Resolution level
                or granularity of text extraction. Defaults to "block".
        """
        # Validate device and resolution values
        if not isinstance(self.encoder, SentenceTransformer):
            raise ValueError("Encoder must be an instance of SentenceTransformer.")
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be either 'cpu', 'cuda', or 'mps'.")
        if self.resolution not in ["block", "page"]:
            raise ValueError("Resolution must be either 'block' or 'page'.")
        
        self.encoder = encoder
        self.device = device
        self.resolution = resolution
    
    def generate_vectordb(
        self, files: List[Dict[str, List[Path]]]
    ) -> VectorDB:
        """
        Generate a vector database for multiple firms' PDF files.

        Args:
            * files (`List[Dict[str, List[Path]]]`): A list of dictionaries, each
                mapping corpus names (e.g. firm names) to lists of PDF file paths.

        Returns:
            `List[Dict[str, Union[str, List[Document]]]]`: List of dictionaries
                containing the firm name and their corresponding vector database.
        """
        vectordb_entries = []
        for corpus in tqdm(files, desc="Generating VectorDB", unit="corpus"):
            for corpus_name, paths in corpus.items():
                temp = self._generate_vectordb_from_file(files=paths)
                entry = VectorDBEntry(corpus_name=corpus_name, documents=temp)
                vectordb_entries.append(entry)
        return VectorDB(entries=vectordb_entries)

    def _generate_vectordb_from_file(
        self,
        files: List[Path],
        preprocess: bool = True,
        normalise_embeddings: bool = True
    ) -> List[Dict[str, Union[str, List[Document]]]]:
        """
        Generate a vector database from a list of PDF files.

        Args:
            * files (`List[Path]`): List of paths to PDF files.
            * resolution (`Literal["block", "page"]`, optional): Resolution level
                for text extraction. Defaults to "block".
            * preprocess (`bool`, optional): Whether to preprocess the text.
                Defaults to True.
            * normalise_embeddings (`bool`, optional): Normalise embeddings to
                unit length. Defaults to True.

        Returns:
            `List[Dict[str, Union[str, List[Document]]]]`: List of dictionaries containing the source and LangChain documents.
        """
        if not files:
            raise ValueError("Files list cannot be empty.")

        docs = []

        for pdf_path in tqdm(
            files, desc="Extracting text", unit="PDF", total=len(files)
        ):
            if not pdf_path.is_file():
                logger.error(f"File not found: {pdf_path}")
                continue

            store = {"source": pdf_path.stem, "contents": []}
            
            # Attempt to open the PDF file, handle exceptions if any issues arise
            try:
                doc = pymupdf.open(filename=str(pdf_path))
            except Exception as e:
                logger.error(f"Error opening {pdf_path}: {e}")
                continue
            
            if self.resolution == "block":
                # Process text blocks within each page as separate documents
                self._process_blocks(
                    doc, store, preprocess, normalise_embeddings
                )
            else:
                # Process each individual page as a single document
                self._process_pages(
                    doc, store, preprocess, normalise_embeddings
                )

            docs.append(store)

        return docs

    def _process_blocks(
        self,
        doc: pymupdf.Document,
        store: Dict[str, Union[str, List[Document]]],
        preprocess: bool = True,
        normalise_embeddings: bool = True
    ) -> None:
        """
        Process text blocks within each page.

        Args:
            * doc (`fitz.Document`): PDF document to process.
            * store (`Dict[str, Union[str, List[Document]]]`): Dictionary to store
                the source and documents.
            * preprocess (`bool`): Whether to preprocess the text. Defaults to True.
            * normalise_embeddings (`bool`): Normalise embeddings to unit length.
                Defaults to True.
        """
        for i, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                text = block[4].strip()
                if text:
                    self._generate_vectordb_entry(
                        text=text,
                        contents=store["contents"],
                        page_num=i,
                        preprocess=preprocess,
                        normalise_embeddings=normalise_embeddings
                    )

    def _process_pages(
        self,
        doc: pymupdf.Document,
        store: Dict[str, Union[str, List[Document]]],
        preprocess: bool = True,
        normalise_embeddings: bool = True
    ) -> None:
        """
        Process entire pages of the PDF.

        Args:
            * doc (`fitz.Document`): PDF document to process.
            * store (`Dict[str, Union[str, List[Document]]]`): Dictionary to store
                the source and documents.
            * preprocess (`bool`): Whether to preprocess the text. Defaults to True.
            * normalise_embeddings (`bool`): Normalise embeddings to unit length.
                Defaults to True.
        """
        for page_num in range(len(doc)):
            text = doc.get_page_text(pno=page_num).strip()
            if text:
                self._generate_vectordb_entry(
                    text=text,
                    contents=store["contents"],
                    page_num=(page_num+1),
                    preprocess=preprocess,
                    normalise_embeddings=normalise_embeddings
                )

    def _generate_vectordb_entry(
        self,
        text: str,
        contents: list,
        page_num: int,
        preprocess: bool = True,
        normalise_embeddings: bool = True
    ) -> None:
        """
        Process individual text block or page and generate embeddings.

        Args:
            * text (`str`): Text to process.
            * contents (`list`): List to store the generated documents.
            * page_num (`int`): Page number in the PDF.
            * preprocess (`bool`): Whether to preprocess the text. Defaults to True.
            * normalise_embeddings (`bool`): Normalise embeddings to unit length.
                Defaults to True.
        """
        try:
            processed_text = preprocess_text(text) if preprocess else text
            embedding = self.encoder.encode(
                sentences=processed_text,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
                normalize_embeddings=normalise_embeddings
            )
            
            # Create a LangChain Document with the processed text and its metadata
            document = Document(
                page_content=processed_text,
                metadata={
                    "page": page_num,
                    "embedding": embedding,
                    "doc_uuid": uuid.uuid5(
                        namespace=uuid.NAMESPACE_DNS, name=processed_text
                    )
                }
            )
            contents.append(document)
        except Exception as e:
            logger.error(f"Error processing text block/page: {e}")