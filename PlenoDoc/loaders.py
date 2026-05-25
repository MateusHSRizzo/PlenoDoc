"""
loaders.py — Carregadores de documentos para o pipeline RAG.

Formatos suportados: PDF, CSV, TXT, DOCX, URL (web scraping)
"""

import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    WebBaseLoader,
)

logger = logging.getLogger("plenodoc.loaders")


def leitura_pdf(caminho: str) -> list:
    """Carrega PDF página por página."""
    return PyPDFLoader(caminho).load()


def leitura_csv(caminho: str) -> list:
    """Carrega CSV como documento único."""
    return UnstructuredCSVLoader(caminho, mode="single").load()


def leitura_txt(caminho: str) -> list:
    """Carrega TXT com fallback de encoding latin-1 para arquivos legados."""
    try:
        return TextLoader(caminho, encoding="utf-8").load()
    except UnicodeDecodeError:
        logger.warning("UTF-8 falhou em %s — usando latin-1.", caminho)
        return TextLoader(caminho, encoding="latin-1").load()


def leitura_docx(caminho: str) -> list:
    """Carrega documento Word (.docx)."""
    return UnstructuredWordDocumentLoader(caminho).load()


def leitura_url(url: str) -> list:
    """
    Carrega conteúdo de uma URL via web scraping (BeautifulSoup + requests).
    Normaliza a metadata 'source' para a URL original.
    """
    loader = WebBaseLoader(
        web_paths=[url],
        requests_kwargs={"timeout": 20, "verify": True},
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url          # garante URL como fonte
        doc.metadata["tipo_fonte"] = "url"
    return docs
