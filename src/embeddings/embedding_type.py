from enum import Enum


class EmbeddingType(str, Enum):
    SENTENCE_TRANSFORMER = "Sentence Transformer"
    OPENAI_EMBEDDING_SMALL = "OpenAI Embedding Small"
    OPENAI_EMBEDDING_LARGE = "OpenAI Embedding Large"
    OPENAI_EMBEDDING_ADA = "OpenAI Embedding Ada"
