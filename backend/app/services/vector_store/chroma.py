from typing import List, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
import chromadb 
from app.core.config import settings
from langchain.schema import BaseRetriever, Document
from .base import BaseVectorStore


from langchain.schema import BaseRetriever, Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import List

from langchain.schema import BaseRetriever, Document
from typing import List

from typing import List
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field

class StaticListRetriever(BaseRetriever, BaseModel):
    docs: List[Document] = Field(...)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

    class Config:
        arbitrary_types_allowed = True



class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation"""
    
    def __init__(self, collection_name: str, embedding_function: Embeddings, **kwargs):
        """Initialize Chroma vector store"""
        # chroma_client = chromadb.HttpClient(
        #      host="http://chromadb:8000"  # hoặc "http://chromadb:8000" nếu backend chạy trong Docker network cùng chromadb container
        # )     

        chroma_client = chromadb.HttpClient(
             host="http://localhost:8001"  # hoặc "http://chromadb:8000" nếu backend chạy trong Docker network cùng chromadb container
        ) 

        self._store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        self._bm25_retriever = None  # Placeholder for BM25 retriever

        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Chroma"""
        self._store.add_documents(documents)
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from Chroma"""
        self._store.delete(ids)
    
    def as_retriever(self, **kwargs: Any):
        """Return a retriever interface"""
        return self._store.as_retriever(**kwargs)
    
    def similarity_search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """Search for similar documents in Chroma"""
        return self._store.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """Search for similar documents in Chroma with score"""
        return self._store.similarity_search_with_score(query, k=k, **kwargs)

    def build_bm25_retriever(self, k: int = 10) -> None:
        """Build BM25 retriever from current documents in the collection."""
        raw_docs = self._store.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
        ]
        self._bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def hybrid_search(self, query: str, k: int = 10, weights: List[float] = [0.4, 0.6]) -> List[Document]:
        """
        Perform hybrid search combining Chroma vector similarity and BM25 keyword search,
        using LangChain's EnsembleRetriever.

        Args:
            query: Query string.
            k: Number of documents to return.
            weights: List of weights for [vector, bm25] retrievers.

        Returns:
            List of retrieved Document objects.
        """
        # Build BM25 retriever if not available
        self.build_bm25_retriever(k=k)

        vector_retriever = self._store.as_retriever(search_kwargs={"k": k})
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, self._bm25_retriever],
            weights=weights,
        )

        return ensemble.invoke(query)

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self._store._client.delete_collection(self._store._collection.name) 