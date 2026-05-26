"""
conftest.py — Fixtures compartilhadas entre todos os testes.
"""
import os, sys, pytest
from pathlib import Path
from unittest.mock import MagicMock

# Garante que o diretório raiz esteja no path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Streamlit mock ────────────────────────────────────────────────────────
# Evita ImportError ao importar módulos que usam st.session_state fora do app
import streamlit as st

@pytest.fixture(autouse=True)
def streamlit_session_state(monkeypatch):
    """Injeta um session_state simples para testes unitários."""
    state = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)
    # Métodos auxiliares usados pelos módulos
    monkeypatch.setattr(st, "cache_resource", lambda **kw: (lambda f: f), raising=False)
    monkeypatch.setattr(st, "warning",  lambda *a,**kw: None, raising=False)
    monkeypatch.setattr(st, "error",    lambda *a,**kw: None, raising=False)
    monkeypatch.setattr(st, "success",  lambda *a,**kw: None, raising=False)
    monkeypatch.setattr(st, "info",     lambda *a,**kw: None, raising=False)
    monkeypatch.setattr(st, "progress", lambda *a,**kw: MagicMock(), raising=False)
    return state


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redireciona .pleno_cache para diretório temporário."""
    import audit, cache_llm, memory, database
    for mod in (audit, cache_llm, memory):
        monkeypatch.setattr(mod, str(list(vars(mod).keys())[0]).split(".")[0], tmp_path, raising=False)
    cache_llm.CACHE_DB = tmp_path / "llm_cache.db"
    audit.AUDIT_DB     = tmp_path / "audit.db"
    memory.MEMORY_DB   = tmp_path / "memory.db"
    database._CACHE_DIR = tmp_path
    database._CONN_FILE = tmp_path / "connections.json"
    database._KEY_FILE  = tmp_path / "db.key"
    return tmp_path


@pytest.fixture
def tmp_users(tmp_path, monkeypatch):
    """Redireciona users.json para diretório temporário."""
    import auth
    monkeypatch.setattr(auth, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    return tmp_path
