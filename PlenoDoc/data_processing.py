"""
data_processing.py — Pipeline RAG: indexação, busca híbrida e reranking.

Melhorias:
  - Busca híbrida BM25 (keyword) + semântica MMR (EnsembleRetriever)
  - Reranking com CrossEncoder (ContextualCompressionRetriever)
  - Scores de relevância nas fontes
  - Resumo automático de documentos no upload
  - Deduplicação de chunks por MD5
  - Metadados de fonte normalizados
  - Suporte a URLs
"""
from __future__ import annotations

import os, hashlib, shutil, json, logging
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from loaders import leitura_pdf, leitura_csv, leitura_txt, leitura_docx, leitura_url
from config import get_settings

logger = logging.getLogger("plenodoc.data")

_LOADERS = {".pdf": leitura_pdf, ".csv": leitura_csv, ".txt": leitura_txt, ".docx": leitura_docx}
_SUMMARIES_FILE = ".pleno_cache/summaries.json"


# ============================================================================
# MODELOS EM CACHE
# ============================================================================

@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def carregar_embeddings() -> HuggingFaceEmbeddings:
    cfg = get_settings()
    logger.info("Carregando embeddings: all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Carregando CrossEncoder…")
def _get_reranker():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    logger.info("Carregando CrossEncoder reranker…")
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================================
# RETRIEVER AVANÇADO
# ============================================================================

def _criar_retriever(vetores: FAISS):
    """
    Monta retriever em camadas:
      1. EnsembleRetriever: BM25 (keyword) + FAISS-MMR (semântico)
      2. ContextualCompressionRetriever com CrossEncoderReranker
    """
    cfg = get_settings()

    # Camada 1 — BM25 + Semântico
    if cfg.enable_hybrid_search:
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            all_docs = list(vetores.docstore._dict.values())
            bm25     = BM25Retriever.from_documents(all_docs, k=cfg.retriever_k)
            semantic = vetores.as_retriever(
                search_type="mmr",
                search_kwargs={"k": cfg.retriever_k, "fetch_k": cfg.retriever_k * 3, "lambda_mult": 0.7},
            )
            base = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.4, 0.6])
            logger.info("Retriever híbrido BM25+MMR criado (%d docs).", len(all_docs))
        except Exception as e:
            logger.warning("Busca híbrida indisponível (%s) — usando semântica pura.", e)
            base = vetores.as_retriever(search_type="mmr", search_kwargs={"k": cfg.retriever_k})
    else:
        base = vetores.as_retriever(search_type="similarity", search_kwargs={"k": cfg.retriever_k})

    # Camada 2 — CrossEncoder Reranking
    if cfg.enable_reranking:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker

            compressor = CrossEncoderReranker(model=_get_reranker(), top_n=cfg.reranker_top_n)
            retriever  = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
            logger.info("CrossEncoder reranking habilitado (top_n=%d).", cfg.reranker_top_n)
            return retriever
        except Exception as e:
            logger.warning("Reranking indisponível (%s) — usando sem reranking.", e)

    return base


# ============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ============================================================================

def carregar_documentos(caminhos: list[str]) -> tuple[list, list[str]]:
    docs, erros = [], []
    for caminho in caminhos:
        ext = os.path.splitext(caminho)[1].lower()
        fn  = _LOADERS.get(ext)
        if fn is None:
            erros.append(f"Formato não suportado: {os.path.basename(caminho)}")
            continue
        try:
            carregados = fn(caminho)
            for doc in carregados:
                doc.metadata["source"]     = os.path.basename(caminho)
                doc.metadata["tipo_fonte"] = "arquivo"
            docs.extend(carregados)
        except Exception as e:
            erros.append(f"Erro em '{os.path.basename(caminho)}': {e}")
            logger.error("Falha ao carregar %s: %s", caminho, e)
    return docs, erros


def _fracionar(documentos: list) -> list:
    cfg = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    ).split_documents(documentos)


def _deduplicar(chunks: list) -> tuple[list, int]:
    seen, unique = set(), []
    for c in chunks:
        h = hashlib.md5(c.page_content.encode("utf-8", errors="replace")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info("Deduplicação: %d chunk(s) duplicados removidos.", removed)
    return unique, removed


# ============================================================================
# RESUMO AUTOMÁTICO
# ============================================================================

def _carregar_summaries() -> dict:
    try:
        if os.path.exists(_SUMMARIES_FILE):
            return json.loads(open(_SUMMARIES_FILE, encoding="utf-8").read())
    except Exception:
        pass
    return {}


def _salvar_summaries(sums: dict) -> None:
    os.makedirs(os.path.dirname(_SUMMARIES_FILE), exist_ok=True)
    with open(_SUMMARIES_FILE, "w", encoding="utf-8") as f:
        json.dump(sums, f, ensure_ascii=False, indent=2)


def gerar_resumo_documento(nome: str, chunks: list) -> str:
    """Gera resumo de 2 frases usando o LLM da sessão (se disponível)."""
    if not get_settings().enable_auto_summary:
        return ""
    llm = st.session_state.get("llm")
    if not llm or not chunks:
        return ""
    amostra = "\n".join(c.page_content for c in chunks[:3])[:2000]
    try:
        resp = llm.invoke([
            SystemMessage(content="Resuma o documento em exatamente 2 frases objetivas em português."),
            HumanMessage(content=f"Documento: {nome}\n\nConteúdo:\n{amostra}"),
        ])
        resumo = resp.content.strip()
        sums   = _carregar_summaries()
        sums[nome] = resumo
        _salvar_summaries(sums)
        logger.info("Resumo gerado para '%s'.", nome)
        return resumo
    except Exception as e:
        logger.warning("Falha ao gerar resumo para '%s': %s", nome, e)
        return ""


def obter_resumos() -> dict:
    return _carregar_summaries()


# ============================================================================
# PERSISTÊNCIA DO RETRIEVER
# ============================================================================

def _salvar_e_atualizar_retriever(vetores: FAISS) -> None:
    cfg = get_settings()
    vetores.save_local(cfg.faiss_path)
    st.session_state.retriever = _criar_retriever(vetores)
    logger.info("Índice FAISS salvo e retriever atualizado.")


# ============================================================================
# API PÚBLICA — ARQUIVOS
# ============================================================================

def adicionar_ao_indice(caminhos: list[str]) -> tuple[bool, str]:
    if not caminhos:
        return False, "Nenhum arquivo fornecido."
    cfg  = get_settings()
    docs, erros = carregar_documentos(caminhos)
    if not docs:
        return False, f"Falha ao ler documentos. Erros: {'; '.join(erros)}"
    chunks, n_dup = _deduplicar(_fracionar(docs))
    embeddings    = carregar_embeddings()
    index_path    = os.path.join(cfg.faiss_path, "index.faiss")
    try:
        if os.path.exists(index_path):
            vetores = FAISS.load_local(cfg.faiss_path, embeddings, allow_dangerous_deserialization=True)
            vetores.add_documents(chunks)
        else:
            vetores = FAISS.from_documents(chunks, embeddings)
        _salvar_e_atualizar_retriever(vetores)
        # Gera resumos em background para novos arquivos
        for caminho in caminhos:
            nome = os.path.basename(caminho)
            if nome not in _carregar_summaries():
                chunks_doc = [c for c in chunks if c.metadata.get("source") == nome]
                gerar_resumo_documento(nome, chunks_doc)
        dedup = f" ({n_dup} duplicatas removidas)" if n_dup else ""
        aviso = f" | ⚠️ {'; '.join(erros)}" if erros else ""
        return True, f"✅ {len(caminhos)} arquivo(s) — {len(chunks)} chunks{dedup}.{aviso}"
    except Exception as e:
        logger.error("Erro ao gravar FAISS: %s", e)
        return False, f"Erro ao gravar o índice: {e}"


def reconstruir_indice(caminhos: list[str]) -> tuple[bool, str]:
    cfg = get_settings()
    if os.path.exists(cfg.faiss_path):
        shutil.rmtree(cfg.faiss_path)
    if not caminhos:
        st.session_state.retriever = None
        return True, "Índice removido — repositório vazio."
    docs, erros = carregar_documentos(caminhos)
    if not docs:
        st.session_state.retriever = None
        return False, f"Nenhum documento válido. Erros: {'; '.join(erros)}"
    try:
        chunks, n_dup = _deduplicar(_fracionar(docs))
        vetores       = FAISS.from_documents(chunks, carregar_embeddings())
        _salvar_e_atualizar_retriever(vetores)
        return True, f"✅ Reconstruído — {len(caminhos)} doc(s), {len(chunks)} chunks."
    except Exception as e:
        st.session_state.retriever = None
        return False, f"Erro: {e}"


# ============================================================================
# API PÚBLICA — URLS
# ============================================================================

def adicionar_url_ao_indice(url: str) -> tuple[bool, str]:
    try:
        docs = leitura_url(url)
    except Exception as e:
        return False, f"Erro ao carregar URL: {e}"
    if not docs:
        return False, "Nenhum conteúdo extraído da URL."
    cfg = get_settings()
    chunks, n_dup = _deduplicar(_fracionar(docs))
    embeddings    = carregar_embeddings()
    index_path    = os.path.join(cfg.faiss_path, "index.faiss")
    try:
        if os.path.exists(index_path):
            vetores = FAISS.load_local(cfg.faiss_path, embeddings, allow_dangerous_deserialization=True)
            vetores.add_documents(chunks)
        else:
            vetores = FAISS.from_documents(chunks, embeddings)
        _salvar_e_atualizar_retriever(vetores)
        dedup = f" ({n_dup} duplicatas)" if n_dup else ""
        return True, f"✅ URL indexada — {len(chunks)} chunks{dedup}."
    except Exception as e:
        return False, f"Erro ao indexar: {e}"


# ============================================================================
# INICIALIZAÇÃO DO RETRIEVER (boot / reload)
# ============================================================================

def inicializar_retriever():
    if st.session_state.get("retriever"):
        return st.session_state.retriever
    cfg = get_settings()
    if not os.path.exists(os.path.join(cfg.faiss_path, "index.faiss")):
        return None
    try:
        vetores = FAISS.load_local(cfg.faiss_path, carregar_embeddings(), allow_dangerous_deserialization=True)
        st.session_state.retriever = _criar_retriever(vetores)
        logger.info("Retriever carregado do disco.")
        return st.session_state.retriever
    except Exception as e:
        logger.error("Falha ao carregar retriever: %s", e)
        return None
