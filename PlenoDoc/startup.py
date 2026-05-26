"""
startup.py — Inicialização automática no Streamlit Cloud.

Garante que o FAISS index seja construído a partir dos documentos
em dados_docs/ caso o filesystem tenha sido resetado (reboot no Cloud).
Também garante que a estrutura de pastas necessária exista.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger("plenodoc.startup")

_DIRS = ["dados_docs", "faiss_index", "logs", ".pleno_cache", "config", ".streamlit"]


def garantir_diretorios() -> None:
    for d in _DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def auto_indexar_se_necessario() -> None:
    """
    Se dados_docs/ tem arquivos mas faiss_index/ não existe (filesystem resetado),
    reconstrói o índice automaticamente ao iniciar.
    """
    faiss_ok   = os.path.exists(os.path.join("faiss_index", "index.faiss"))
    docs_path  = "dados_docs"
    docs_exist = os.path.exists(docs_path) and bool([
        f for f in os.listdir(docs_path)
        if not f.startswith(".") and os.path.isfile(os.path.join(docs_path, f))
    ])

    if docs_exist and not faiss_ok:
        logger.info("FAISS ausente mas dados_docs/ tem arquivos — reconstruindo índice…")
        try:
            import streamlit as st
            from data_processing import reconstruir_indice
            caminhos = [
                os.path.join(docs_path, f)
                for f in os.listdir(docs_path)
                if not f.startswith(".")
            ]
            ok, msg = reconstruir_indice(caminhos)
            if ok:
                logger.info("Índice reconstruído com sucesso no boot: %s", msg)
            else:
                logger.warning("Falha ao reconstruir índice no boot: %s", msg)
        except Exception as e:
            logger.error("Erro no auto-index de boot: %s", e)
    elif not docs_exist:
        logger.info("dados_docs/ vazio — aguardando upload de documentos.")
    else:
        logger.info("FAISS index já existe — nenhuma ação necessária.")


def run() -> None:
    garantir_diretorios()
    auto_indexar_se_necessario()
