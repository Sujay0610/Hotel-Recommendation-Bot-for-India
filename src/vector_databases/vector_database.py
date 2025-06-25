import os
from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            OpenAIEmbeddings)
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from abc import abstractmethod
from typing import Union
import chromadb

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')


class VectorDatabase:
    """
    The base class for vector database that stores the documents for retriever
    tool.
    """
    @classmethod
    @abstractmethod
    def get(cls,
            embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
            persist_directory: str,
            collection_name: str):
        """
        Retrieve the vector database.
        """
        pass

    @classmethod
    @abstractmethod
    def _create_db(cls,
                   embedding_model: Union[HuggingFaceEmbeddings,
                                          OpenAIEmbeddings],
                   persist_directory: str,
                   collection_name: str):
        pass

    @classmethod
    @abstractmethod
    def _load_db(cls, embedding_model: Union[HuggingFaceEmbeddings,
                                             OpenAIEmbeddings],
                 persist_directory: str,
                 collection_name: str):
        pass

    @staticmethod
    def _check_path_exist(path) -> bool:
        """
        Check whether the path already exists and if it already has content.

        Parameters
        ----------
        path: str
            The path (directory) to be checked.

        Returns
        -------
        bool
            If True, the path already exists, and it already has content inside.
        """
        if os.path.exists(path):
            return len(os.listdir(path)) > 0
        else:
            return False


class ChromaDB(VectorDatabase):
    """
    An implementation for Chroma vector database.
    """
    @classmethod
    def get(cls,
            embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
            persist_directory: str,
            collection_name: str) -> Chroma:
        """
        Retrieve the vector database.

        Parameters
        ----------
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The embedding model.
        persist_directory: str
            The directory where the ChromaDB is persisted.
        collection_name: str
            The name of the collection in ChromaDB.

        Returns
        -------
        Chroma
            The Chroma database.
        """
        print(f"Loading ChromaDB from {persist_directory} with collection '{collection_name}'")
        return cls._load_db(embedding_model, persist_directory, collection_name)

    @classmethod
    def _create_db(cls,
                   embedding_model: Union[HuggingFaceEmbeddings,
                                          OpenAIEmbeddings],
                   persist_directory: str,
                   collection_name: str):
        raise NotImplementedError("ChromaDB creation is handled by prepare_chroma_db.py")

    @classmethod
    def _load_db(cls,
                 embedding_model: Union[HuggingFaceEmbeddings,
                                        OpenAIEmbeddings],
                 persist_directory: str,
                 collection_name: str) -> Chroma:
        """
        To load the vector database.

        Parameters
        ----------
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The embedding model.
        persist_directory: str
            The directory where the ChromaDB is persisted.
        collection_name: str
            The name of the collection in ChromaDB.

        Returns
        -------
        Chroma
            The Chroma vector database.
        """
        client = chromadb.PersistentClient(path=persist_directory)
        langchain_chroma = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        return langchain_chroma


class FaissDB(VectorDatabase):
    """
    An implementation for FAISS database.
    """
    FAISS_DB_PATH = os.path.join(DATA_DIR, "faiss_db")

    @classmethod
    def get(cls, embedding_model, persist_directory, collection_name):
        pass

    @classmethod
    def _create_db(cls,
                   embedding_model,
                   persist_directory,
                   collection_name):
        pass

    @classmethod
    def _load_db(cls,
                 embedding_model: Union[HuggingFaceEmbeddings,
                                        OpenAIEmbeddings],
                 persist_directory: str,
                 collection_name: str):
        pass
