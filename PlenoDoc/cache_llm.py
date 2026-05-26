"""
cache_llm.py — Cache SQLite de respostas do LLM.
Evita rechamar a API para perguntas idênticas dentro do TTL configurado.
"""
from __future__ import annotations
import hashlib, sqlite3, time, logging
from pathlib import Path
from config import get_settings

logger   = logging.getLogger("plenodoc.cache")
CACHE_DB = Path(".pleno_cache/llm_cache.db")


def _init() -> None:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key      TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            model    TEXT NOT NULL DEFAULT '',
            ts       REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _chave(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()


def _ttl() -> float:
    return get_settings().llm_cache_ttl_hours * 3600


def get_cached(prompt: str, model: str) -> str | None:
    if not get_settings().enable_llm_cache:
        return None
    _init()
    chave = _chave(prompt, model)
    conn  = sqlite3.connect(str(CACHE_DB))
    row   = conn.execute("SELECT response, ts FROM cache WHERE key=?", (chave,)).fetchone()
    conn.close()
    if row and (time.time() - row[1]) < _ttl():
        logger.debug("Cache HIT: %s…", prompt[:50])
        return row[0]
    return None


def set_cached(prompt: str, model: str, response: str) -> None:
    if not get_settings().enable_llm_cache:
        return
    _init()
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, response, model, ts) VALUES (?,?,?,?)",
        (_chave(prompt, model), response, model, time.time()),
    )
    conn.commit()
    conn.close()


def limpar_expirado() -> int:
    _init()
    conn = sqlite3.connect(str(CACHE_DB))
    cur  = conn.execute("DELETE FROM cache WHERE ts < ?", (time.time() - _ttl(),))
    conn.commit()
    n = cur.rowcount
    conn.close()
    if n:
        logger.info("Cache: %d entrada(s) expirada(s) removidas.", n)
    return n


def stats() -> dict:
    _init()
    conn   = sqlite3.connect(str(CACHE_DB))
    total  = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    valido = conn.execute("SELECT COUNT(*) FROM cache WHERE ts > ?", (time.time() - _ttl(),)).fetchone()[0]
    conn.close()
    return {"total": total, "validos": valido, "expirados": total - valido}
