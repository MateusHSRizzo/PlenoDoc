# data_processing.py

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from loaders import leitura_pdf, leitura_csv, leitura_txt, leitura_docx

# --- Constantes de Dados e Processamento ---
CAMINHO_DOCUMENTOS = "dados_docs"
CAMINHO_FAISS = "faiss_index"
NOME_MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

def carregar_documentos(caminhos_arquivos):
    """
    Carrega documentos a partir de uma lista de caminhos de arquivos.
    """
    documentos = []
    for caminho in caminhos_arquivos:
        nome_arquivo = os.path.basename(caminho)
        try:
            if nome_arquivo.endswith('.pdf'): documentos.extend(leitura_pdf(caminho))
            elif nome_arquivo.endswith('.csv'): documentos.extend(leitura_csv(caminho))
            elif nome_arquivo.endswith('.txt'): documentos.extend(leitura_txt(caminho))
            elif nome_arquivo.endswith('.docx'): documentos.extend(leitura_docx(caminho))
            else: st.warning(f'Formato não suportado: {nome_arquivo}')
        except Exception as e: st.error(f"Erro ao carregar {nome_arquivo}: {e}")
    return documentos

def fracionar_documentos(documentos):
    """
    Divide os documentos em pedaços (chunks) menores para vetorização.
    """
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return divisor.split_documents(documentos)

def atualizar_vetores(documentos_a_processar):
    """
    Cria ou atualiza o índice vetorial FAISS de forma incremental.
    """
    if not documentos_a_processar:
        st.info("Nenhum documento para processar.")
        # Se a base estiver vazia, apaga o índice para evitar erros de leitura futuros
        if os.path.exists(CAMINHO_FAISS):
            import shutil
            shutil.rmtree(CAMINHO_FAISS)
        return

    documentos_divididos = fracionar_documentos(documentos_a_processar)
    embeddings = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDINGS, model_kwargs={'device': 'cpu'})
    
    # Verifica a existência do arquivo de índice, não apenas da pasta.
    caminho_indice_faiss = os.path.join(CAMINHO_FAISS, "index.faiss")

    if os.path.exists(caminho_indice_faiss):
        st.write("Adicionando/Atualizando documentos no índice existente...")
        vetores_amz = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
        vetores_amz.add_documents(documentos_divididos)
    else:
        st.write("Criando um novo índice de vetores...")
        vetores_amz = FAISS.from_documents(documentos_divididos, embeddings)
    
    vetores_amz.save_local(CAMINHO_FAISS)
    st.session_state.retriever = vetores_amz.as_retriever()
    st.success("Base de conhecimento atualizada com sucesso!")

def inicializar_retriever():
    """
    Carrega o retriever FAISS do disco se ele já tiver sido criado.
    """
    if st.session_state.retriever:
        return st.session_state.retriever
    
    if os.path.exists(os.path.join(CAMINHO_FAISS, "index.faiss")):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDINGS, model_kwargs={'device': 'cpu'})
            vetores_amz = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
            st.session_state.retriever = vetores_amz.as_retriever()
            return st.session_state.retriever
        except Exception as e:
            st.error(f"Não foi possível carregar a base de conhecimento: {e}")
            return None
    return None