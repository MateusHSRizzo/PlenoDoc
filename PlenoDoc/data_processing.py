import os
import shutil
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from loaders import leitura_pdf, leitura_csv, leitura_txt, leitura_docx

CAMINHO_DOCUMENTOS = "dados_docs"
CAMINHO_FAISS = "faiss_index"
NOME_MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

@st.cache_resource
def carregar_embeddings():
    return HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDINGS, model_kwargs={'device': 'cpu'})

def carregar_documentos(caminhos_arquivos):
    documentos = []
    erros = []
    for caminho in caminhos_arquivos:
        nome_arquivo = os.path.basename(caminho)
        try:
            if nome_arquivo.endswith('.pdf'): documentos.extend(leitura_pdf(caminho))
            elif nome_arquivo.endswith('.csv'): documentos.extend(leitura_csv(caminho))
            elif nome_arquivo.endswith('.txt'): documentos.extend(leitura_txt(caminho))
            elif nome_arquivo.endswith('.docx'): documentos.extend(leitura_docx(caminho))
            else: erros.append(f'Formato não suportado: {nome_arquivo}')
        except Exception as e: 
            erros.append(f"Erro ao carregar {nome_arquivo}: {e}")
    return documentos, erros

def fracionar_documentos(documentos):
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return divisor.split_documents(documentos)

def adicionar_ao_indice(caminhos_arquivos):
    if not caminhos_arquivos:
        return False, "Nenhum arquivo fornecido."

    documentos, erros = carregar_documentos(caminhos_arquivos)
    if not documentos:
        return False, f"Falha ao ler documentos. Erros: {erros}"

    documentos_divididos = fracionar_documentos(documentos)
    embeddings = carregar_embeddings()
    caminho_indice_faiss = os.path.join(CAMINHO_FAISS, "index.faiss")

    if os.path.exists(caminho_indice_faiss):
        vetores_amz = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
        vetores_amz.add_documents(documentos_divididos)
    else:
        vetores_amz = FAISS.from_documents(documentos_divididos, embeddings)
    
    vetores_amz.save_local(CAMINHO_FAISS)
    return True, "Documentos indexados com sucesso."

def reconstruir_indice(caminhos_arquivos):
    if os.path.exists(CAMINHO_FAISS):
        shutil.rmtree(CAMINHO_FAISS)

    if not caminhos_arquivos:
        return True, "Índice limpo. Nenhum documento restante."

    documentos, erros = carregar_documentos(caminhos_arquivos)
    if not documentos:
        return False, "Nenhum documento válido para reconstrução."

    documentos_divididos = fracionar_documentos(documentos)
    embeddings = carregar_embeddings()
    vetores_amz = FAISS.from_documents(documentos_divididos, embeddings)
    vetores_amz.save_local(CAMINHO_FAISS)
    
    return True, "Índice reconstruído com sucesso."

def inicializar_retriever():
    if os.path.exists(os.path.join(CAMINHO_FAISS, "index.faiss")):
        try:
            embeddings = carregar_embeddings()
            vetores_amz = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
            return vetores_amz.as_retriever(search_kwargs={"k": 4})
        except Exception:
            return None
    return None
