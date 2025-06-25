from langchain_community.embeddings import HuggingFaceEmbeddings
from .embedding_type import EmbeddingType
from typing import Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

REGISTRY_EMBEDDING = {
    EmbeddingType.SENTENCE_TRANSFORMER: "all-MiniLM-L6-v2",
    EmbeddingType.OPENAI_EMBEDDING_SMALL: "text-embedding-3-small",
    EmbeddingType.OPENAI_EMBEDDING_LARGE: "text-embedding-3-large",
    EmbeddingType.OPENAI_EMBEDDING_ADA: "text-embedding-ada-002",
}


class Embeddings:
    """
    A class to get the embedding model.
    """
    @classmethod
    def get(cls, embedding_name: EmbeddingType) -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings]:

        if embedding_name == EmbeddingType.SENTENCE_TRANSFORMER:
            print(f"[INFO] Using {EmbeddingType.SENTENCE_TRANSFORMER}")
            return HuggingFaceEmbeddings(
                model_name=REGISTRY_EMBEDDING[embedding_name]
            )
        elif embedding_name in [
            EmbeddingType.OPENAI_EMBEDDING_SMALL,
            EmbeddingType.OPENAI_EMBEDDING_LARGE,
            EmbeddingType.OPENAI_EMBEDDING_ADA,
        ]:
            print(f"[INFO] Using {embedding_name}")
            return OpenAIEmbeddings(
                model=REGISTRY_EMBEDDING[embedding_name]
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_name}.")

