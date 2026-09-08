"""结构感知切分：MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter。

对应设计文档 §5.2 ②，标题层级写入 metadata.title_path 保证语义块完整。
"""
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 按文档 §5.2 配置
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120


def get_markdown_header_splitter() -> MarkdownHeaderTextSplitter:
    """按 #/##/### 标题切分，标题层级写入 metadata。"""
    return MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS)


def get_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """子块再走递归字符切分。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )


def split_document(text: str, base_metadata: dict | None = None) -> list[dict]:
    """对一个文档内容做「标题切分 → 递归切分」两级处理。

    返回 [{text, title, section, ...metadata}] 列表。
    """
    base_metadata = base_metadata or {}
    md_splitter = get_markdown_header_splitter()
    sub_splitter = get_recursive_splitter()

    rows: list[dict] = []
    for sec in md_splitter.split_text(text):
        title = sec.metadata.get("h1", "")
        section = sec.metadata.get("h2", "")
        for chunk in sub_splitter.split_text(sec.page_content):
            rows.append(
                {
                    "text": chunk,
                    "title": title,
                    "section": section,
                    **base_metadata,
                }
            )
    return rows
