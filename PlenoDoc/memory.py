"""
memory.py — Memória persistente de conversas entre sessões (SQLite).
Carregada no login e salva a cada mensagem.
"""
from __future__ import annotations
import sqlite3, logging
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

logger    = logging.getLogger("plenodoc.memory")
MEMORY_DB = Path(".pleno_cache/memory.db")
MAX_MSGS  = 100   # mensagens por usuário retidas em disco


def _init() -> None:
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historico (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL,
            role     TEXT    NOT NULL,
            content  TEXT    NOT NULL,
            ts       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def salvar_mensagem(username: str, role: str, content: str) -> None:
    """role = 'human' | 'ai'"""
    _init()
    try:
        conn = sqlite3.connect(str(MEMORY_DB))
        conn.execute(
            "INSERT INTO historico (username, role, content) VALUES (?,?,?)",
            (username, role, content),
        )
        # Mantém apenas as MAX_MSGS mais recentes por usuário
        conn.execute("""
            DELETE FROM historico WHERE id IN (
                SELECT id FROM historico WHERE username=?
                ORDER BY id DESC LIMIT -1 OFFSET ?
            )
        """, (username, MAX_MSGS))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Falha ao salvar memória: %s", e)


def carregar_historico(username: str) -> list:
    """Retorna lista de HumanMessage / AIMessage na ordem cronológica."""
    _init()
    try:
        conn = sqlite3.connect(str(MEMORY_DB))
        rows = conn.execute(
            "SELECT role, content FROM historico WHERE username=? ORDER BY id DESC LIMIT ?",
            (username, MAX_MSGS),
        ).fetchall()
        conn.close()
        msgs = []
        for role, content in reversed(rows):
            msgs.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
        return msgs
    except Exception as e:
        logger.error("Falha ao carregar memória: %s", e)
        return []


def limpar_historico(username: str) -> None:
    _init()
    conn = sqlite3.connect(str(MEMORY_DB))
    conn.execute("DELETE FROM historico WHERE username=?", (username,))
    conn.commit()
    conn.close()
    logger.info("Histórico limpo para: %s", username)


def contar_mensagens(username: str) -> int:
    _init()
    conn = sqlite3.connect(str(MEMORY_DB))
    n    = conn.execute("SELECT COUNT(*) FROM historico WHERE username=?", (username,)).fetchone()[0]
    conn.close()
    return n
