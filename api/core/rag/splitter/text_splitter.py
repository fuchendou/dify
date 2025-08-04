from __future__ import annotations

import copy
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Sequence, Set
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from core.rag.models.document import BaseDocumentTransformer, Document

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({re.escape(separator)})", text)
            splits = [_splits[i - 1] + _splits[i] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 != 0:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if (s not in {"", "\n"})]


class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[list[str]], list[int]] = lambda x: [len(x) for x in x],
        keep_separator: bool = False,
        add_start_index: bool = False,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components."""

    def create_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata or {})
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: list[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str, lengths: list[int]) -> list[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function([separator])[0]

        docs = []
        current_doc: list[str] = []
        total = 0
        index = 0
        for d in splits:
            _len = lengths[index]
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0
                    ):
                        total -= self._length_function([current_doc[0]])[0] + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
            index += 1
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase  # type: ignore

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError("Tokenizer received was not an instance of PreTrainedTokenizerBase")

            def _huggingface_tokenizer_length(text: str) -> int:
                return len(tokenizer.encode(text))

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. Please install it with `pip install transformers`."
            )
        return cls(length_function=lambda x: [_huggingface_tokenizer_length(text) for text in x], **kwargs)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        raise NotImplementedError



# @dataclass(frozen=True, kw_only=True, slots=True)
@dataclass(frozen=True)
class Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[list[int]], str]
    encode: Callable[[str], list[int]]


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], Set[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> list[str]:
        def _encode(_text: str) -> list[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: Optional[list[str]] = None,
        protected_tags: Optional[list[str]] = None,
        keep_separator: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        # protected tag list
        self._protected_tags = protected_tags or ["image", "table"]
        # construct protected tag pattern
        tags_pattern = "|".join(self._protected_tags)
        self._block_pattern = re.compile(rf"(<(?:{tags_pattern})>.*?</(?:{tags_pattern})>)", re.DOTALL)

    def _presplit_blocks(self, text: str) -> list[str]:
        # split protected blocks and common chunks
        parts = self._block_pattern.split(text)
        return [p for p in parts if p]
        
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        for i, sep in enumerate(separators):
            if sep == "" or re.search(re.escape(sep), text):
                chosen, rest = sep, separators[i+1:]
                break
        splits = _split_text_with_regex(text, chosen, self._keep_separator)
        good, good_lens, result = [], [], []
        for s in splits:
            length = self._length_function([s])[0]
            if length < self._chunk_size:
                good.append(s)
                good_lens.append(length)
            else:
                if good:
                    result.extend(self._merge_splits(good, "" if self._keep_separator else chosen, good_lens))
                    good, good_lens = [], []
                if rest:
                    result.extend(self._recursive_split(s, rest))
                else:
                    result.append(s)
        if good:
            result.extend(self._merge_splits(good, "" if self._keep_separator else chosen, good_lens))
        return result

    def split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        # prechunk according to the protected blocks
        segments = self._presplit_blocks(text)
        for seg in segments:
            if self._block_pattern.fullmatch(seg):
                chunks.append(seg)
            else:
                chunks.extend(self._recursive_split(seg, self._separators))
        # postprocess: merge protected blocks with previous chunk if it's short
        merged: list[str] = []
        cursor = 0
        for chunk in chunks:
            is_prot = self._block_pattern.fullmatch(chunk)
            start = text.find(chunk, cursor)
            # judge if it's a protected block in the middle of a chunk
            inline_prot = False
            if is_prot and start > 0:
                prev_char = text[start - 1]
                # inline protected block
                if prev_char not in ['\n']:
                    inline_prot = True
            if inline_prot and merged:
                # if the previous chunk is short, merge with it
                if len(merged[-1]) < self._chunk_size:
                    merged[-1] = merged[-1] + chunk
                    cursor = start + len(chunk)
                    continue
            # otherwise, append it as a new chunk
            merged.append(chunk)
            cursor = start + len(chunk)

        return merged

class SequentialTextSplitter(TextSplitter):
    """Splitting text by sequential characters with protected block support."""

    def __init__(
        self,
        read_size: int = 1024,
        protected_tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new SequentialTextSplitter.
        
        Args:
            read_size: Size of buffer to read at each step
            protected_tags: List of tags to treat as protected blocks (e.g., ["image", "table"])
            **kwargs: Additional arguments passed to parent TextSplitter
        """
        super().__init__(**kwargs)
        self._read_size = read_size
        self._protected_tags = protected_tags or ["image", "table"]
        # Sort tags by length in descending order for better matching
        self._protected_tags = sorted(self._protected_tags, key=len, reverse=True)
        # construct protected tag patterns
        self._protected_patterns = {}
        for tag in self._protected_tags:
            self._protected_patterns[tag] = {
                'open': f"<{tag}>",
                'close': f"</{tag}>"
            }

    def _try_match_protected_tag(self, text: str, pos: int) -> Optional[tuple[str, int]]:
        """Try to match a protected tag starting at position pos.
        
        Optimized version using more efficient matching strategy.
        
        Returns:
            Tuple of (matched_block, block_length) if found, None otherwise
        """
        if pos >= len(text) or text[pos] != '<':
            return None
        
        # Pre-calculate remaining text to avoid repeated slicing
        remaining_text = text[pos:]
        
        # Tags are already sorted by length in descending order
        for tag in self._protected_tags:
            open_tag = self._protected_patterns[tag]['open']
            close_tag = self._protected_patterns[tag]['close']
            
            # Check if we have an opening tag at current position
            if remaining_text.startswith(open_tag):
                # Find the corresponding closing tag
                close_pos = text.find(close_tag, pos + len(open_tag))
                if close_pos != -1:
                    # Found complete protected block
                    block_end = close_pos + len(close_tag)
                    protected_block = text[pos:block_end]
                    return protected_block, len(protected_block)
        
        return None

    def _get_overlap_content(self, chunk_content: str, overlap_size: int) -> str:
        """Get overlap content from the end of a chunk."""
        if overlap_size <= 0 or not chunk_content:
            return ""
        
        if len(chunk_content) <= overlap_size:
            return chunk_content
        
        return chunk_content[-overlap_size:]

    def _finalize_current_chunk(self, current_chunk_parts: list[str], chunks: list[str]) -> tuple[list[str], int]:
        """Finalize current chunk and prepare for next chunk with overlap.
        
        Args:
            current_chunk_parts: List of parts that make up the current chunk
            chunks: List to append the finalized chunk to
            
        Returns:
            Tuple of (new_chunk_parts, new_size) for the next chunk
        """
        if not current_chunk_parts:
            return [], 0
        
        # Efficient chunk content building
        chunk_content = self._build_chunk_content(current_chunk_parts)
        chunks.append(chunk_content)
        
        # Get overlap content from the end of current chunk
        overlap_content = self._get_overlap_content(chunk_content, self._chunk_overlap)
        new_chunk_parts = [overlap_content] if overlap_content else []
        new_size = len(overlap_content)
        
        return new_chunk_parts, new_size

    def _build_chunk_content(self, chunk_parts: list[str]) -> str:
        """Build chunk content efficiently based on the number of parts."""
        if not chunk_parts:
            return ""
        elif len(chunk_parts) == 1:
            return chunk_parts[0]
        else:
            return ''.join(chunk_parts)

    def _is_chunk_start_position(self, current_chunk_parts: list[str], current_size: int) -> bool:
        """Check if current position is suitable for starting a new chunk.
        
        Args:
            current_chunk_parts: Current chunk parts
            current_size: Current chunk size
            
        Returns:
            True if this is a good position to start a new chunk
        """
        if not current_chunk_parts:
            return True
        
        # If only one element and it's overlap content
        if len(current_chunk_parts) == 1:
            return current_size <= self._chunk_overlap
        
        return False

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using sequential scanning with protected block support.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks, each respecting chunk_size limits except when 
            containing protected blocks which are kept intact.
            
        Note:
            Protected blocks will never be split and may cause chunks to exceed
            the configured chunk_size limit.
        """
        if not text:
            return []
        
        chunks = []
        
        # 1. Initialize
        pos = 0
        current_chunk_parts = []  # Renamed for clarity
        current_size = 0
        
        # 2. Main loop: check if pos reaches the end of text
        while pos < len(text):
            # 3. Read buffer
            buffer_end = min(pos + self._read_size, len(text))
            buffer = text[pos:buffer_end]
            i = 0  # Buffer local pointer
            
            # 4. Buffer sequential scan
            while i < len(buffer):
                current_pos_in_text = pos + i

                # 4.2 If buffer[i] == '<': try to match protected tag
                if buffer[i] == '<':
                    # Try to match protected tag (may need to extend beyond current buffer)
                    protected_match = self._try_match_protected_tag(text, current_pos_in_text)
                    
                    if protected_match:
                        protected_block, block_length = protected_match
                        
                        # Check if it starts from protected block using optimized method
                        is_chunk_start = self._is_chunk_start_position(current_chunk_parts, current_size)
                        
                        # If current chunk reaches size limit and not from protected block, finalize current chunk
                        if current_size >= self._chunk_size and current_chunk_parts and not is_chunk_start:
                            current_chunk_parts, current_size = self._finalize_current_chunk(current_chunk_parts, chunks)
                        
                        # Add protected block to current chunk (allow exceeding chunk_size)
                        current_chunk_parts.append(protected_block)
                        current_size += block_length
                        i += block_length
                        
                        # If protected block exceeds current buffer, need to adjust pos
                        if current_pos_in_text + block_length > buffer_end:
                            pos = current_pos_in_text + block_length
                            break  # Break out of buffer scan, enter next window
                    else:
                        # 4.2 Not match protected tag, handle as normal character
                        # 5. Check current chunk size
                        if current_size + 1 > self._chunk_size and current_chunk_parts:
                            current_chunk_parts, current_size = self._finalize_current_chunk(current_chunk_parts, chunks)
                        
                        current_chunk_parts.append(buffer[i])
                        current_size += 1
                        i += 1
                else:
                    # 4.1 Handle normal character
                    # 5. Check current chunk size
                    if current_size + 1 > self._chunk_size and current_chunk_parts:
                        current_chunk_parts, current_size = self._finalize_current_chunk(current_chunk_parts, chunks)
                    
                    current_chunk_parts.append(buffer[i])
                    current_size += 1
                    i += 1
            
            # 7. Advance global pointer (if not early break due to protected block)
            if i >= len(buffer):
                pos += i
        
        # 8. All text processed
        if current_chunk_parts:
            chunk_content = self._build_chunk_content(current_chunk_parts)
            if chunk_content.strip():  # Only add non-empty chunk
                chunks.append(chunk_content)
        
        return chunks

class ChunkTextSplitter(TextSplitter):
    """Splitting text by fixed chunk size with protected block support."""

    def __init__(
        self,
        protected_tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new ChunkTextSplitter.
        
        Args:
            protected_tags: List of tags to treat as protected blocks (e.g., ["image", "table"])
            **kwargs: Additional arguments passed to parent TextSplitter
        """
        super().__init__(**kwargs)
        self._protected_tags = protected_tags or ["image", "table"]
        # Sort tags by length in descending order for better matching
        self._protected_tags = sorted(self._protected_tags, key=len, reverse=True)

    def _initial_split(self, text: str) -> list[str]:
        """Split text by fixed chunk size.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks, each with length <= chunk_size
        """
        if not text:
            return []
        
        chunks = []
        for i in range(0, len(text), self._chunk_size):
            chunk = text[i:i + self._chunk_size]
            chunks.append(chunk)
        
        return chunks

    def _find_tag_in_chunk(self, chunk: str, tag: str) -> tuple[bool, bool]:
        """check if the chunk contains the open or close tag of the protected block"""
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        has_open = open_tag in chunk
        has_close = close_tag in chunk
        
        return has_open, has_close

    def _find_matching_close_tag(self, chunks: list[str], start_idx: int, tag: str) -> int:
        """
        backward search for the matching close tag
        """
        close_tag = f"</{tag}>"
        
        for i in range(start_idx, len(chunks)):
            if close_tag in chunks[i]:
                return i
        
        return -1

    def _find_matching_open_tag(self, chunks: list[str], end_idx: int, tag: str) -> int:
        """
        forward search for the matching open tag
        """
        open_tag = f"<{tag}>"
        
        for i in range(end_idx, -1, -1):
            if open_tag in chunks[i]:
                return i
        
        return -1

    def _merge_chunks(self, chunks: list[str], start_idx: int, end_idx: int) -> str:
        """
        merge the chunks within the specified range
        """
        if start_idx > end_idx or start_idx < 0 or end_idx >= len(chunks):
            return ""
        
        return "".join(chunks[start_idx:end_idx + 1])

    def _check_boundary_tags(self, chunks: list[str], idx: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """
        检查 chunk 边界的残留保护块标记，处理标签在任意位置被分割的情况
        只检查相邻的 chunk，因为标签前后缀只会出现在相邻的 chunk 中
        
        Returns:
            Tuple of (tag_name, start_merge_idx, end_merge_idx) if merge is needed, otherwise (None, None, None)
        """
        if idx >= len(chunks):
            return None, None, None
        
        current_chunk = chunks[idx]
        
        # 检查当前 chunk 结尾是否包含开始标签的部分
        if idx + 1 < len(chunks):
            next_chunk = chunks[idx + 1]
            
            # 遍历所有保护标签
            for tag in self._protected_tags:
                open_tag = f"<{tag}>"
                
                # 检查开始标签是否被分割在两个 chunk 之间
                # 尝试所有可能的分割位置
                for split_pos in range(1, len(open_tag)):
                    prefix = open_tag[:split_pos]  # 标签前缀部分
                    suffix = open_tag[split_pos:]  # 标签后缀部分
                    
                    # 检查当前 chunk 是否以前缀结尾，下一个 chunk 是否以后缀开头
                    if current_chunk.endswith(prefix) and next_chunk.startswith(suffix):
                        # 找到匹配的结束标记
                        close_idx = self._find_matching_close_tag(chunks, idx + 1, tag)
                        if close_idx != -1:
                            return tag, idx, close_idx
        
        # 检查当前 chunk 开头是否包含结束标签的部分
        if idx > 0:
            prev_chunk = chunks[idx - 1]
            
            # 遍历所有保护标签
            for tag in self._protected_tags:
                close_tag = f"</{tag}>"
                
                # 检查结束标签是否被分割在两个 chunk 之间
                # 尝试所有可能的分割位置
                for split_pos in range(1, len(close_tag)):
                    prefix = close_tag[:split_pos]  # 标签前缀部分
                    suffix = close_tag[split_pos:]  # 标签后缀部分
                    
                    # 检查上一个 chunk 是否以前缀结尾，当前 chunk 是否以后缀开头
                    if prev_chunk.endswith(prefix) and current_chunk.startswith(suffix):
                        # 找到匹配的开始标记
                        open_idx = self._find_matching_open_tag(chunks, idx - 1, tag)
                        if open_idx != -1:
                            return tag, open_idx, idx
        
        return None, None, None

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using fixed chunk size with protected block support.
        
        按照指定的处理流程：
        1. 初始准备：将文本按照固定 chunk_size 切分
        2. 主循环：遍历每一个 chunk，检查保护块
        3. 特殊字符检查：处理 chunk 边界残留的保护块标记
        """
        if not text:
            return []
        
        # 1. 初始准备
        chunks = self._initial_split(text)
        if not chunks:
            return []
        
        # 2. 主循环：遍历每一个 chunk
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            merged = False
            
            # 3. 当前 chunk 检查是否存在保护块头尾标记
            for tag in self._protected_tags:
                has_open, has_close = self._find_tag_in_chunk(current_chunk, tag)
                
                # 3.2 如果发现 <tag>（保护块起始）
                if has_open and not has_close:
                    close_idx = self._find_matching_close_tag(chunks, i + 1, tag)
                    if close_idx != -1:
                        # 合并所有涉及的 chunks
                        merged_content = self._merge_chunks(chunks, i, close_idx)
                        chunks[i] = merged_content
                        
                        # 删除被合并的后续 chunks
                        for _ in range(close_idx - i):
                            if i + 1 < len(chunks):
                                chunks.pop(i + 1)
                        
                        merged = True
                        break
                
                # 3.3 如果发现 </tag>（保护块结尾）但没有对应起始 <tag>
                elif has_close and not has_open:
                    open_idx = self._find_matching_open_tag(chunks, i - 1, tag)
                    if open_idx != -1:
                        # 合并所有涉及的 chunks
                        merged_content = self._merge_chunks(chunks, open_idx, i)
                        chunks[open_idx] = merged_content
                        
                        # 删除被合并的后续 chunks（包括当前 chunk）
                        for _ in range(i - open_idx):
                            if open_idx + 1 < len(chunks):
                                chunks.pop(open_idx + 1)
                        
                        # 调整索引，因为当前 chunk 被合并到前面了
                        i = open_idx
                        merged = True
                        break
            
            # 4. 特殊字符检查：chunk 边界残留保护块标记
            if not merged:
                tag, start_idx, end_idx = self._check_boundary_tags(chunks, i)
                if tag is not None and start_idx is not None and end_idx is not None:
                    # 合并涉及的 chunks
                    merged_content = self._merge_chunks(chunks, start_idx, end_idx)
                    chunks[start_idx] = merged_content
                    
                    # 删除被合并的后续 chunks
                    for _ in range(end_idx - start_idx):
                        if start_idx + 1 < len(chunks):
                            chunks.pop(start_idx + 1)
                    
                    # 调整索引
                    i = start_idx
                    merged = True
            
            # 5. 指针推进
            i += 1
        
        # 6. 过滤空 chunks 并返回结果
        return [chunk for chunk in chunks if chunk.strip()]