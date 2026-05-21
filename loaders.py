from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

def leitura_pdf(caminho):
    return PyPDFLoader(caminho).load()

def leitura_csv(caminho):
    return UnstructuredCSVLoader(caminho, mode="single").load()

def leitura_txt(caminho):
    return TextLoader(caminho).load()

def leitura_docx(caminho):
    return UnstructuredWordDocumentLoader(caminho).load()