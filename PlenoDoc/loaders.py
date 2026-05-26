"""
loaders.py — Carregadores de documentos para o pipeline RAG.

Formatos: PDF (com fallback OCR), CSV, TXT, DOCX, URL
"""
import logging
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredCSVLoader,
    UnstructuredWordDocumentLoader, TextLoader, WebBaseLoader,
)

logger = logging.getLogger("plenodoc.loaders")

# ── PDF ───────────────────────────────────────────────────────────────────

def _pdf_via_ocr(caminho: str) -> list:
    """Fallback OCR para PDFs escaneados (sem camada de texto)."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from langchain_core.documents import Document

        logger.info("OCR iniciado para '%s'…", caminho)
        imagens = convert_from_path(caminho, dpi=200)
        docs    = []
        for i, img in enumerate(imagens, start=1):
            texto = pytesseract.image_to_string(img, lang="por+eng")
            if texto.strip():
                docs.append(Document(
                    page_content=texto,
                    metadata={"source": caminho, "page": i, "ocr": True},
                ))
        logger.info("OCR concluído: %d página(s) extraída(s).", len(docs))
        return docs
    except ImportError:
        logger.warning("pdf2image/pytesseract não instalados — OCR indisponível.")
        return []
    except Exception as e:
        logger.warning("OCR falhou em '%s': %s", caminho, e)
        return []


def leitura_pdf(caminho: str) -> list:
    """Carrega PDF; usa OCR automaticamente se o texto extraído for insuficiente."""
    docs = PyPDFLoader(caminho).load()
    conteudo_total = sum(len(d.page_content.strip()) for d in docs)
    if conteudo_total < 100:                      # provável PDF escaneado
        logger.info("'%s' parece escaneado (texto=%d chars). Tentando OCR…", caminho, conteudo_total)
        docs_ocr = _pdf_via_ocr(caminho)
        if docs_ocr:
            return docs_ocr
    return docs

# ── CSV ──────────────────────────────────────────────────────────────────

def leitura_csv(caminho: str) -> list:
    return UnstructuredCSVLoader(caminho, mode="single").load()

# ── TXT ──────────────────────────────────────────────────────────────────

def leitura_txt(caminho: str) -> list:
    """UTF-8 com fallback para latin-1."""
    try:
        return TextLoader(caminho, encoding="utf-8").load()
    except UnicodeDecodeError:
        logger.warning("UTF-8 falhou em '%s' — usando latin-1.", caminho)
        return TextLoader(caminho, encoding="latin-1").load()

# ── DOCX ─────────────────────────────────────────────────────────────────

def leitura_docx(caminho: str) -> list:
    return UnstructuredWordDocumentLoader(caminho).load()

# ── URL ───────────────────────────────────────────────────────────────────

def leitura_url(url: str) -> list:
    """Web scraping com BeautifulSoup. Normaliza metadata source para a URL."""
    loader = WebBaseLoader(
        web_paths=[url],
        requests_kwargs={"timeout": 20, "verify": True},
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"]     = url
        doc.metadata["tipo_fonte"] = "url"
    return docs
