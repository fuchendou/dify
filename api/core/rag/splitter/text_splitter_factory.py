"""Factory for creating and managing text splitters."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from core.model_manager import ModelInstance
from core.rag.splitter.fixed_text_splitter import (
    FixedRecursiveCharacterTextSplitter,
)
from core.rag.splitter.markdown_header_text_splitter import MarkdownHeaderTextSplitter
from core.rag.splitter.text_splitter import (
    TextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger(__name__)


class SplitterType(Enum):
    """Enumeration of available text splitter types."""
    
    FIXED_RECURSIVE = "fixed_recursive"
    MARKDOWN = "markdown"
    TOKEN = "token"
    AUTO_DETECT = "auto_detect"


class TextSplitterFactory:
    """Factory class for creating text splitters based on content type and requirements."""
    
    @staticmethod
    def create_splitter(
        splitter_type: SplitterType,
        chunk_size: int = 4000,
        chunk_overlap: int = 0,
        fixed_separator: str = "\n\n",
        separators: Optional[list[str]] = None,
        embedding_model_instance: Optional[ModelInstance] = None,
        **kwargs: Any,
    ) -> TextSplitter:
        """
        Create a text splitter based on the specified type.
        
        Args:
            splitter_type: Type of splitter to create
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks
            fixed_separator: Fixed separator for splitting
            separators: List of separators to use
            embedding_model_instance: Model instance for token counting
            **kwargs: Additional arguments for specific splitters
            
        Returns:
            TextSplitter: The created text splitter instance
            
        Raises:
            ValueError: If an unsupported splitter type is specified
        """
        logger.info(f"Creating text splitter of type: {splitter_type.value}")
        
        if splitter_type == SplitterType.FIXED_RECURSIVE:
            return FixedRecursiveCharacterTextSplitter.from_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                fixed_separator=fixed_separator,
                separators=separators or ["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance,
                **kwargs
            )
        
        elif splitter_type == SplitterType.MARKDOWN:
            return MarkdownHeaderTextSplitter.from_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                fixed_separator=fixed_separator,
                separators=separators,
                embedding_model_instance=embedding_model_instance,
                **kwargs
            )
        
        elif splitter_type == SplitterType.TOKEN:
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
        elif splitter_type == SplitterType.AUTO_DETECT:
            return TextSplitterFactory._create_auto_detect_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                fixed_separator=fixed_separator,
                separators=separators,
                embedding_model_instance=embedding_model_instance,
                **kwargs
            )
    
    @staticmethod
    def _create_auto_detect_splitter(
        chunk_size: int,
        chunk_overlap: int,
        fixed_separator: str,
        separators: Optional[list[str]],
        embedding_model_instance: Optional[ModelInstance],
        **kwargs: Any,
    ) -> TextSplitter:
        """
        Create an auto-detecting splitter that chooses the appropriate splitter based on content.
        """
        class AutoDetectTextSplitter(TextSplitter):
            def __init__(self):
                super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                self.markdown_splitter = TextSplitterFactory.create_splitter(
                    SplitterType.MARKDOWN,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    fixed_separator=fixed_separator,
                    separators=separators,
                    embedding_model_instance=embedding_model_instance,
                    **kwargs
                )
                self.default_splitter = TextSplitterFactory.create_splitter(
                    SplitterType.FIXED_RECURSIVE,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    fixed_separator=fixed_separator,
                    separators=separators,
                    embedding_model_instance=embedding_model_instance,
                    **kwargs
                )
            
            def split_text(self, text: str) -> list[str]:
                if MarkdownHeaderTextSplitter.is_markdown_content(text):
                    logger.info("Detected Markdown content, using markdown splitter")
                    return self.markdown_splitter.split_text(text)
                else:
                    logger.info("Using default splitter for regular content")
                    return self.default_splitter.split_text(text)
        
        return AutoDetectTextSplitter()
    
    @staticmethod
    def get_splitter_for_processing_rule(
        processing_rule_mode: str,
        max_tokens: int,
        chunk_overlap: int,
        separator: str,
        embedding_model_instance: Optional[ModelInstance],
        auto_detect_markdown: bool = True,
    ) -> TextSplitter:
        """
        Get the appropriate splitter based on processing rule mode.
        
        Args:
            processing_rule_mode: Processing rule mode ("custom", "hierarchical", etc.)
            max_tokens: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
            separator: Separator to use
            embedding_model_instance: Model instance for token counting
            auto_detect_markdown: Whether to auto-detect markdown content
            
        Returns:
            TextSplitter: The appropriate text splitter
        """
        if separator:
            separator = separator.replace("\\n", "\n")
        
        if processing_rule_mode in ["custom", "hierarchical"]:
            if auto_detect_markdown:
                return TextSplitterFactory.create_splitter(
                    SplitterType.AUTO_DETECT,
                    chunk_size=max_tokens,
                    chunk_overlap=chunk_overlap,
                    fixed_separator=separator,
                    separators=["\n\n", "。", ". ", " ", ""],
                    embedding_model_instance=embedding_model_instance,
                )
            else:
                return TextSplitterFactory.create_splitter(
                    SplitterType.FIXED_RECURSIVE,
                    chunk_size=max_tokens,
                    chunk_overlap=chunk_overlap,
                    fixed_separator=separator,
                    separators=["\n\n", "。", ". ", " ", ""],
                    embedding_model_instance=embedding_model_instance,
                )
        else:
            # Automatic segmentation
            from models.dataset import DatasetProcessRule
            
            return TextSplitterFactory.create_splitter(
                SplitterType.TOKEN,
                chunk_size=DatasetProcessRule.AUTOMATIC_RULES["segmentation"]["max_tokens"],
                chunk_overlap=DatasetProcessRule.AUTOMATIC_RULES["segmentation"]["chunk_overlap"],
                separators=["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance,
            )
    
    @staticmethod
    def detect_content_type(text: str) -> SplitterType:
        """
        Detect the content type and recommend an appropriate splitter.
        
        Args:
            text: Text content to analyze
            
        Returns:
            SplitterType: Recommended splitter type
        """
        if MarkdownHeaderTextSplitter.is_markdown_content(text):
            return SplitterType.MARKDOWN
        
        return SplitterType.FIXED_RECURSIVE