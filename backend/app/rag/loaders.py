"""多源文档加载：Markdown / PDF / 网页。

对应设计文档 §5.2 ① Loader 步骤。
"""
from pathlib import Path

from langchain_core.documents import Document


def load_markdown(path: str | Path) -> list[Document]:
    """加载 Markdown 文档。

    设计取舍：不引入 UnstructuredMarkdownLoader——它强依赖 unstructured 全家桶
    （Windows 上安装易失败，且对纯 md 解析属于杀鸡用牛刀）。Markdown 的标题结构
    由后续 MarkdownHeaderTextSplitter 重新解析，这里只需把文件读成原文 Document。
    """
    fp = Path(path)
    text = fp.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(fp)})]


def load_pdf(path: str | Path) -> list[Document]:
    """加载 PDF 文档。"""
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def load_webpage(url: str) -> list[Document]:
    """抓取公开网页页面。"""
    from langchain_community.document_loaders import WebBaseLoader

    return WebBaseLoader(url).load()


def load_directory(
    root: str | Path,
    city: str,
    category: str,
) -> list[Document]:
    """遍历 knowledge_base/{city}/{category}/*.md。

    返回 Document 列表，city/category/source 已注入 metadata
    （文件名即语料标题，由 splitter 解析 h1 时写入 title）。
    """
    root = Path(root) / city / category
    if not root.exists():
        return []
    out: list[Document] = []
    for fp in root.glob("*.md"):
        for doc in load_markdown(fp):
            doc.metadata["source"] = str(fp)
            doc.metadata["city"] = city
            doc.metadata["category"] = category
            out.append(doc)
    return out
