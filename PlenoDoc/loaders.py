from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

def leitura_pdf(caminho):
    """Carrega documentos PDF."""
    return PyPDFLoader(caminho).load()

def leitura_csv(caminho):
    """Carrega documentos CSV."""
    return UnstructuredCSVLoader(caminho, mode="single").load()

def leitura_txt(caminho):
    """Carrega documentos de texto."""
    return TextLoader(caminho).load()

def leitura_docx(caminho):
    """Carrega documentos DOCX."""

    return UnstructuredWordDocumentLoader(caminho).load()
