"""
data_processing.py — Pipeline de vetorização e indexação de documentos.

Melhorias aplicadas:
  - Deduplicação de chunks por hash MD5 do conteúdo
  - Metadados de fonte normalizados (source = nome do arquivo ou URL)
  - Suporte a indexação de URLs via leitura_url()
  - Logging estruturado em vez de st.write
"""

import os
import hashlib
import logging
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from loaders import leitura_pdf, leitura_csv, leitura_txt, leitura_docx, leitura_url

logger = logging.getLogger("plenodoc.data")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CAMINHO_DOCUMENTOS   = "dados_docs"
CAMINHO_FAISS        = "faiss_index"
NOME_MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

_EXTENSOES_SUPORTADAS = {".pdf", ".csv", ".txt", ".docx"}
_LOADERS = {
    ".pdf":  leitura_pdf,
    ".csv":  leitura_csv,
    ".txt":  leitura_txt,
    ".docx": leitura_docx,
}


# ---------------------------------------------------------------------------
# Embeddings — singleton em cache (carregado uma única vez por processo)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def carregar_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Carregando modelo de embeddings: %s", NOME_MODELO_EMBEDDINGS)
    return HuggingFaceEmbeddings(
        model_name=NOME_MODELO_EMBEDDINGS,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Carregamento de documentos
# ---------------------------------------------------------------------------

def carregar_documentos(caminhos: list[str]) -> tuple[list, list[str]]:
    """
    Carrega documentos de uma lista de caminhos de arquivo.
    Normaliza a metadata 'source' para apenas o nome do arquivo.
    Retorna (documentos, lista_de_erros).
    """
    docs, erros = [], []
    for caminho in caminhos:
        ext = os.path.splitext(caminho)[1].lower()
        fn  = _LOADERS.get(ext)
        if fn is None:
            erros.append(f"Formato não suportado: {os.path.basename(caminho)}")
            continue
        try:
            carregados = fn(caminho)
            # Normaliza source para nome do arquivo (sem caminho completo)
            for doc in carregados:
                doc.metadata["source"]     = os.path.basename(caminho)
                doc.metadata["tipo_fonte"] = "arquivo"
            docs.extend(carregados)
            logger.info("Carregado: %s (%d páginas/chunks)", os.path.basename(caminho), len(carregados))
        except Exception as e:
            erros.append(f"Erro em '{os.path.basename(caminho)}': {e}")
            logger.error("Falha ao carregar %s: %s", caminho, e)
    return docs, erros


# ---------------------------------------------------------------------------
# Chunking + deduplicação
# ---------------------------------------------------------------------------

def _fracionar_documentos(documentos: list) -> list:
    """Divide documentos em chunks para vetorização."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documentos)


def _deduplicar_chunks(chunks: list) -> tuple[list, int]:
    """
    Remove chunks com conteúdo idêntico (hash MD5).
    Retorna (chunks_únicos, quantidade_removida).
    """
    seen   = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode("utf-8", errors="replace")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removidos = len(chunks) - len(unique)
    if removidos:
        logger.info("Deduplicação: %d chunks duplicados removidos.", removidos)
    return unique, removidos


# ---------------------------------------------------------------------------
# Persistência do retriever
# ---------------------------------------------------------------------------

def _salvar_e_atualizar_retriever(vetores: FAISS) -> None:
    """Salva o índice FAISS no disco e atualiza o retriever na sessão."""
    vetores.save_local(CAMINHO_FAISS)
    st.session_state.retriever = vetores.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    logger.info("Índice FAISS salvo em '%s'.", CAMINHO_FAISS)


# ---------------------------------------------------------------------------
# API pública — arquivos
# ---------------------------------------------------------------------------

def adicionar_ao_indice(caminhos: list[str]) -> tuple[bool, str]:
    """
    Carrega arquivos, chunka, deduplica e adiciona ao índice FAISS.
    Cria um novo índice se ainda não existir.
    """
    if not caminhos:
        return False, "Nenhum arquivo fornecido."

    docs, erros = carregar_documentos(caminhos)
    if not docs:
        return False, f"Falha ao ler documentos. Erros: {'; '.join(erros)}"

    chunks, n_dup = _deduplicar_chunks(_fracionar_documentos(docs))
    embeddings    = carregar_embeddings()
    index_path    = os.path.join(CAMINHO_FAISS, "index.faiss")

    try:
        if os.path.exists(index_path):
            vetores = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
            vetores.add_documents(chunks)
        else:
            vetores = FAISS.from_documents(chunks, embeddings)

        _salvar_e_atualizar_retriever(vetores)
        dedup_msg = f" ({n_dup} duplicatas removidas)" if n_dup else ""
        aviso_msg = f" | Avisos: {'; '.join(erros)}" if erros else ""
        return True, f"✅ {len(caminhos)} arquivo(s) indexado(s) — {len(chunks)} chunks{dedup_msg}.{aviso_msg}"

    except Exception as e:
        logger.error("Erro ao gravar índice FAISS: %s", e)
        return False, f"Erro ao gravar o índice: {e}"


def reconstruir_indice(caminhos: list[str]) -> tuple[bool, str]:
    """
    Reconstrói o índice FAISS do zero (use após remoção de documentos).
    """
    if os.path.exists(CAMINHO_FAISS):
        shutil.rmtree(CAMINHO_FAISS)
        logger.info("Índice FAISS removido para reconstrução.")

    if not caminhos:
        st.session_state.retriever = None
        return True, "Índice removido — repositório vazio."

    docs, erros = carregar_documentos(caminhos)
    if not docs:
        st.session_state.retriever = None
        return False, f"Nenhum documento válido para reconstrução. Erros: {'; '.join(erros)}"

    try:
        chunks, n_dup = _deduplicar_chunks(_fracionar_documentos(docs))
        embeddings    = carregar_embeddings()
        vetores       = FAISS.from_documents(chunks, embeddings)
        _salvar_e_atualizar_retriever(vetores)
        return True, f"✅ Índice reconstruído — {len(caminhos)} doc(s), {len(chunks)} chunks."
    except Exception as e:
        st.session_state.retriever = None
        logger.error("Erro ao reconstruir índice: %s", e)
        return False, f"Erro ao reconstruir o índice: {e}"


# ---------------------------------------------------------------------------
# API pública — URLs
# ---------------------------------------------------------------------------

def adicionar_url_ao_indice(url: str) -> tuple[bool, str]:
    """
    Faz scraping de uma URL, chunka o conteúdo e adiciona ao índice FAISS.
    """
    try:
        docs = leitura_url(url)
    except Exception as e:
        return False, f"Erro ao carregar URL: {e}"

    if not docs:
        return False, "Nenhum conteúdo extraído da URL."

    chunks, n_dup = _deduplicar_chunks(_fracionar_documentos(docs))
    embeddings    = carregar_embeddings()
    index_path    = os.path.join(CAMINHO_FAISS, "index.faiss")

    try:
        if os.path.exists(index_path):
            vetores = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
            vetores.add_documents(chunks)
        else:
            vetores = FAISS.from_documents(chunks, embeddings)

        _salvar_e_atualizar_retriever(vetores)
        dedup_msg = f" ({n_dup} duplicatas removidas)" if n_dup else ""
        logger.info("URL indexada: %s — %d chunks.", url, len(chunks))
        return True, f"✅ URL indexada — {len(chunks)} chunks{dedup_msg}."
    except Exception as e:
        logger.error("Erro ao indexar URL %s: %s", url, e)
        return False, f"Erro ao indexar: {e}"


# ---------------------------------------------------------------------------
# Inicialização do retriever (carregado do disco)
# ---------------------------------------------------------------------------

def inicializar_retriever():
    """
    Retorna o retriever da sessão ou tenta carregá-lo do disco.
    """
    if st.session_state.get("retriever"):
        return st.session_state.retriever

    index_path = os.path.join(CAMINHO_FAISS, "index.faiss")
    if not os.path.exists(index_path):
        return None

    try:
        embeddings = carregar_embeddings()
        vetores    = FAISS.load_local(CAMINHO_FAISS, embeddings, allow_dangerous_deserialization=True)
        st.session_state.retriever = vetores.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        logger.info("Retriever carregado do disco.")
        return st.session_state.retriever
    except Exception as e:
        logger.error("Falha ao carregar retriever: %s", e)
        return None
