"""Functionality for splitting text."""
from __future__ import annotations

from typing import Any, Optional

from core.model_manager import ModelInstance
from core.model_runtime.model_providers.__base.tokenizers.gpt2_tokenzier import GPT2Tokenizer
from core.rag.splitter.text_splitter import (
    TS,
    Collection,
    Literal,
    RecursiveCharacterTextSplitter,
    ChunkTextSplitter,
    Set,
    TokenTextSplitter,
    Union,
)


class EnhanceRecursiveCharacterTextSplitter(ChunkTextSplitter):
    """
    This class is used to implement from_gpt2_encoder, to prevent using of tiktoken
    """

    @classmethod
    def from_encoder(
        cls: type[TS],
        embedding_model_instance: Optional[ModelInstance],
        allowed_special: Union[Literal["all"], Set[str]] = set(),  # noqa: UP037
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",  # noqa: UP037
        **kwargs: Any,
    ):
        def _token_encoder(texts: list[str]) -> list[int]:
            if not texts:
                return []

            if embedding_model_instance:
                return embedding_model_instance.get_text_embedding_num_tokens(texts=texts)
            else:
                return [GPT2Tokenizer.get_num_tokens(text) for text in texts]

        def _character_encoder(texts: list[str]) -> list[int]:
            if not texts:
                return []

            return [len(text) for text in texts]

        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {
                "model_name": embedding_model_instance.model if embedding_model_instance else "gpt2",
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_character_encoder, **kwargs)


class FixedRecursiveCharacterTextSplitter(EnhanceRecursiveCharacterTextSplitter):
    def __init__(
        self,
        fixed_separator: str = "\n\n",
        separators: Optional[list[str]] = None,
        read_size: int = 1024,
        protected_tags: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._fixed_separator = fixed_separator
        self._separators = separators
        
        self._splitter = ChunkTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        """Split text using conditional strategy based on fixed_separator."""
        if not text:
            return []
        
        return self._splitter.split_text(text)

