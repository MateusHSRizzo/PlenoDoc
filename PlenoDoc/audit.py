"""
audit.py — Registro de auditoria de ações do sistema.
Grava em logs/audit.log (texto) + .pleno_cache/audit.db (SQLite consultável).
"""
from __future__ import annotations
import logging
import sqlite3
import datetime
import pandas as pd
import streamlit as st
from pathlib import Path

logger   = logging.getLogger("plenodoc.audit")
AUDIT_DB = Path(".pleno_cache/audit.db")


def _init() -> None:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT    NOT NULL,
            username TEXT    NOT NULL DEFAULT 'sistema',
            role     TEXT    NOT NULL DEFAULT '',
            acao     TEXT    NOT NULL,
            detalhes TEXT             DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


def registrar(acao: str, detalhes: str = "") -> None:
    """Registra uma ação de auditoria para o usuário logado."""
    username = st.session_state.get("username", "sistema") or "sistema"
    role     = st.session_state.get("user_role", "") or ""
    ts       = datetime.datetime.now().isoformat(timespec="seconds")

    logger.info("[AUDIT] %s | %s (%s) | %s | %s", ts, username, role, acao, detalhes[:120])

    try:
        _init()
        conn = sqlite3.connect(str(AUDIT_DB))
        conn.execute(
            "INSERT INTO audit_log (ts, username, role, acao, detalhes) VALUES (?,?,?,?,?)",
            (ts, username, role, acao, detalhes[:500]),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Falha ao gravar audit: %s", e)


def carregar_log(limit: int = 500) -> list[dict]:
    try:
        _init()
        conn = sqlite3.connect(str(AUDIT_DB))
        rows = conn.execute(
            "SELECT ts, username, role, acao, detalhes FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [{"ts": r[0], "usuário": r[1], "perfil": r[2], "ação": r[3], "detalhes": r[4]} for r in rows]
    except Exception:
        return []


def painel_audit_ui() -> None:
    """Componente Streamlit para visualização do log de auditoria (somente admin)."""
    st.subheader("🔎 Log de Auditoria")
    registros = carregar_log()
    if not registros:
        st.info("Nenhum registro ainda.")
        return

    df = pd.DataFrame(registros)
    busca = st.text_input("🔍 Filtrar por usuário ou ação", key="audit_busca")
    if busca:
        mask = df.apply(lambda row: busca.lower() in str(row).lower(), axis=1)
        df   = df[mask]

    st.caption(f"{len(df)} registro(s)")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Exportar CSV",
        df.to_csv(index=False).encode(),
        "audit_log.csv",
        "text/csv",
        use_container_width=True,
    )
