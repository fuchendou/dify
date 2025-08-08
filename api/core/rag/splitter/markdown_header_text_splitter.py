"""Markdown-specific text splitter implementation."""

from __future__ import annotations

import re
from typing import Any

from core.rag.models.document import Document
from core.rag.splitter.fixed_text_splitter import FixedRecursiveCharacterTextSplitter

class MarkdownHeaderTextSplitter(FixedRecursiveCharacterTextSplitter):
    """
    A text splitter specifically designed for Markdown content.
    It uses Markdown-specific separators to maintain document structure according to the header.
    Inherits from FixedRecursiveCharacterTextSplitter for recursive splitting capability.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        fixed_separator: str = "\n", 
        is_used_FixedRecursiveCharacterTextSplitter: bool = False,
        **kwargs: Any
    ):
        """Create a MarkdownHeaderTextSplitter."""
        super().__init__(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            fixed_separator=fixed_separator,
            **kwargs
        )

        self.headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]
        self.is_used_FixedRecursiveCharacterTextSplitter = is_used_FixedRecursiveCharacterTextSplitter
        self.global_uuid_list = []

    @classmethod
    def from_encoder(
        cls,
        chunk_size,
        chunk_overlap,
        **kwargs: Any,
    ):
        """Create a MarkdownHeaderTextSplitter with encoder-based length function."""

        def _character_encoder(texts: list[str]) -> list[int]:
            if not texts:
                return []
            return [len(text) for text in texts]

        # Remove embedding_model_instance from kwargs if present
        # as it's not needed for MarkdownHeaderTextSplitter
        kwargs.pop('embedding_model_instance', None)

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_character_encoder,
            **kwargs
        )

    @staticmethod
    def is_markdown_content(text: str) -> bool:
        """
        Judge whether the text is markdown content
        Args:
            text: The text to be judged
        Returns:
            bool: If the text is markdown content, return True
        """
        markdown_identifier = "Use markdown textsplitter to process the context"
        return text.strip().startswith(markdown_identifier)
    
    def _extract_block_uuids(self, text: str) -> list[str]:
        """
        Extract all block UUIDs from the text
        Args:
            text: The text to extract UUIDs from
        Returns:
            list[str]: List of extracted UUIDs
        """
        uuids = []
        lines = text.split('\n')

        in_block_info = False

        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped == "# BLOCK_INFORMATION":
                in_block_info = True
                continue
            
            if in_block_info and line_stripped.startswith("# ") and line_stripped != "# BLOCK_INFORMATION":
                break
            
            if in_block_info and line_stripped:
                uuids.append(line_stripped)
        
        return uuids
    
    def _parse_header_line(self, line: str) -> tuple[bool, str, str, int]:
        """
        Parse a line of text to determine if it's a header
        
        Returns:
            tuple: (is_header, header_name, header_text, header_level)
        """
        line_stripped = line.strip()
        
        for header_symbol, header_name in self.headers:
            if line_stripped.startswith(header_symbol + ' '):
                header_level = len(header_symbol)
                header_text = line_stripped[len(header_symbol):].strip()
                return True, header_name, header_text, header_level
        
        return False, "", "", 0

    def _update_header_state(self, current_headers: dict, header_name: str, header_text: str, header_level: int) -> dict:
        """
        Update header state, clear headers at the same level and lower levels
        
        Args:
            current_headers: Current header state
            header_name: New header name
            header_text: New header text
            header_level: New header level
            
        Returns:
            dict: Updated header state
        """
        # Clear headers at the same level and lower levels
        keys_to_remove = []
        for existing_header_symbol, existing_header_name in self.headers:
            if len(existing_header_symbol) >= header_level:
                keys_to_remove.append(existing_header_name)
        
        for key in keys_to_remove:
            current_headers.pop(key, None)
        
        # Set new header
        current_headers[header_name] = header_text
        
        return current_headers

    def _parse_text_to_chunks(self, text: str) -> list[Document]:
        """
        Parse text into a list of chunks with metadata
        
        Args:
            text: Markdown text to parse
            
        Returns:
            list[Document]: List of parsed Documents chunks

        """
        lines = text.split('\n')
        chunks = []
        current_chunk = Document(page_content="")
        current_headers = {}
        
        for line in lines:
            is_header, header_name, header_text, header_level = self._parse_header_line(line)
            
            if is_header:
                # Save current chunk (if it has content)
                if current_chunk.page_content.strip():
                    current_chunk.metadata = current_headers.copy()
                    current_chunk.metadata["is_exist_uuid"] = False
                    chunks.append(current_chunk)
                
                # Update header state
                current_headers = self._update_header_state(
                    current_headers, header_name, header_text, header_level
                )
                
                # Start new chunk
                current_chunk = Document(page_content="")

            else:
                # Add content line
                self._add_content_line(current_chunk, line)
        
        # Save the last chunk
        if current_chunk.page_content.strip():
            current_chunk.metadata = current_headers.copy()
            current_chunk.metadata['is_exist_uuid'] = False
            chunks.append(current_chunk)

        return chunks

    def _add_content_line(self, chunk: Document, line: str):
        """
        Add content line to chunk
        
        Args:
            chunk: Chunk to add content to
            line: Line to add
        """
        line_stripped = line.strip()
        
        if line_stripped:  # Non-empty line
            if chunk.page_content:
                chunk.page_content += '\n' + line
            else:
                chunk.page_content = line

    def _recursive_split_with_metadata(self, chunks: list[Document]) -> list[Document]:
        """
        Apply recursive splitting while preserving metadata
        
        Args:
            chunks: List of Document chunks to be further split
            
        Returns:
            list[Document]: List of recursively split Document chunks with preserved metadata
        """
        final_chunks = []

        for chunk in chunks:
            original_metadata = chunk.metadata.copy()
            # Use parent class's split_text method for recursive splitting
            # chunk_size doesn't consider metadata length, only content length
            split_texts = super().split_text(chunk.page_content)
            # Create new Document chunks with preserved metadata
            for split_text in split_texts:
                new_chunk = Document(page_content=split_text, metadata=original_metadata)
                final_chunks.append(new_chunk)

        return final_chunks

    def _process_uuid_blocks(self, chunks: list[Document]) -> list[Document]:
        """
        process uuid block in chunks
        
        Args:
            chunks: current chunks
            
        Returns:
            list[Document]: processed chunks
        """
        final_chunks = []
        uuid_chunks = []
        
        for chunk in chunks:
            remaining_content = chunk.page_content
            chunk_modified = False
            
            for uuid in self.global_uuid_list[:]:
                pattern = rf'<!--BLOCK_START:{re.escape(uuid)}-->(.*?)<!--BLOCK_END:{re.escape(uuid)}-->'
                match = re.search(pattern, remaining_content, re.DOTALL)
                
                if match:
                    uuid_content = match.group(1).strip()
                    
                    remaining_content = re.sub(pattern, '', remaining_content, flags=re.DOTALL)
                    chunk_modified = True
                    
                    uuid_chunk = Document(
                        page_content=uuid_content,
                        metadata={
                            **chunk.metadata,
                            'is_exist_uuid': True,
                            'uuid': uuid
                        }
                    )
                    uuid_chunks.append(uuid_chunk)
                    self.global_uuid_list.remove(uuid)
            
            if chunk_modified:
                remaining_content = remaining_content.strip()
                if remaining_content:
                    chunk.page_content = remaining_content
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        final_chunks.extend(uuid_chunks)
        
        return final_chunks

    def _chunk_to_string(self, chunk: Document) -> str:
        """
        Combine chunk and metadata into an expressive string
        
        Args:
            chunk: Chunk to convert
            
        Returns:
            str: Formatted string
        """
        if not chunk.metadata:
            return chunk.page_content.strip()
        
        # Build header context information
        header_context = []
        for _, header_name in self.headers:
            if header_name in chunk.metadata:
                header_context.append(f"{header_name}: {chunk.metadata[header_name]}")

        if chunk.metadata.get('is_exist_uuid', False) and 'uuid' in chunk.metadata:
            header_context.append(f"UUID: {chunk.metadata['uuid']}")
        
        # Combine format: [Header Context] + Content
        if header_context:
            context_str = " | ".join(header_context)
            return f"[{context_str}]\n{chunk.page_content.strip()}"
        else:
            return chunk.page_content.strip()

    def split_text(self, text: str) -> list[str]:
        """
        Split Markdown text while preserving structure.
        
        Args:
            text: The Markdown text to split
            
        Returns:
            list[str]: List of text chunks with embedded metadata
        """
        # 1. Extract uuid
        if "# BLOCK_INFORMATION" in text:
            self.global_uuid_list = self._extract_block_uuids(text)

        # 2. Parse text into chunks with metadata
        chunks = self._parse_text_to_chunks(text)

        # 3. Recursive split with metadata
        if self.is_used_FixedRecursiveCharacterTextSplitter:
            chunks = self._recursive_split_with_metadata(chunks)

        # 4. Postprocess uuid blocks
        if self.global_uuid_list:
            chunks = self._process_uuid_blocks(chunks)
        
        # 5. Convert to string list
        result = []
        for chunk in chunks:
            chunk_str = self._chunk_to_string(chunk)
            if chunk_str:
                result.append(chunk_str)
        
        return result