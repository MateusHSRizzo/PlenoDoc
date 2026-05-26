"""
test_auth.py — Testes unitários para auth.py.
Cobre: hashing, verificação, permissões, timeout, criação de usuários.
"""
import time
import pytest
import streamlit as st


# ── Hashing ──────────────────────────────────────────────────────────────────

def test_hash_senha_retorna_hex():
    from auth import _hash_senha
    h, s = _hash_senha("minha_senha")
    assert len(h) == 64      # SHA-256 hex = 64 chars
    assert len(s) == 64      # salt 32 bytes → 64 hex chars


def test_hash_mesma_senha_salt_diferente():
    """Duas chamadas com senhas iguais devem gerar hashes diferentes (salt aleatório)."""
    from auth import _hash_senha
    h1, s1 = _hash_senha("abc123")
    h2, s2 = _hash_senha("abc123")
    assert h1 != h2
    assert s1 != s2


def test_hash_com_salt_fixo_deterministico():
    from auth import _hash_senha
    h1, _ = _hash_senha("abc123", salt_hex="aabbcc" * 10 + "aabb")   # 64 hex chars
    h2, _ = _hash_senha("abc123", salt_hex="aabbcc" * 10 + "aabb")
    assert h1 == h2


def test_verificar_senha_correta():
    from auth import _hash_senha, _verificar_senha
    h, s = _hash_senha("senha_segura")
    assert _verificar_senha("senha_segura", h, s) is True


def test_verificar_senha_errada():
    from auth import _hash_senha, _verificar_senha
    h, s = _hash_senha("senha_certa")
    assert _verificar_senha("senha_errada", h, s) is False


def test_hmac_compare_igual():
    from auth import hmac_compare
    assert hmac_compare("abc", "abc") is True


def test_hmac_compare_diferente():
    from auth import hmac_compare
    assert hmac_compare("abc", "xyz") is False


# ── Permissões ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,acao,esperado", [
    ("admin",    "gerenciar_docs",    True),
    ("admin",    "ver_audit",         True),
    ("operator", "banco_query",       True),
    ("operator", "gerenciar_docs",    False),
    ("viewer",   "chat",              True),
    ("viewer",   "banco_query",       False),
    ("viewer",   "gerenciar_docs",    False),
    ("desconhecido", "chat",          False),
])
def test_tem_permissao(role, acao, esperado, streamlit_session_state):
    from auth import tem_permissao
    streamlit_session_state["user_role"] = role
    assert tem_permissao(acao) is esperado


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout_nao_disparado(streamlit_session_state):
    from auth import verificar_timeout
    streamlit_session_state["logged_in"]    = True
    streamlit_session_state["last_activity"] = time.time()
    assert verificar_timeout() is False


def test_timeout_sessao_nao_logada(streamlit_session_state):
    from auth import verificar_timeout
    streamlit_session_state["logged_in"] = False
    assert verificar_timeout() is False


def test_timeout_expirado(streamlit_session_state, monkeypatch):
    from auth import verificar_timeout
    from config import get_settings
    monkeypatch.setattr(get_settings(), "session_timeout_minutes", 1, raising=False)
    streamlit_session_state["logged_in"]     = True
    streamlit_session_state["last_activity"] = time.time() - 3700   # 1h + 100s atrás
    # Injeta st.warning como no-op
    result = verificar_timeout()
    # Após timeout, logged_in deve ser False
    assert streamlit_session_state.get("logged_in") is False


# ── Criação de usuários ───────────────────────────────────────────────────────

def test_criar_usuario_sucesso(tmp_users):
    from auth import criar_usuario, _carregar_usuarios
    ok, msg = criar_usuario("joao", "senha123", "viewer", "João Silva")
    assert ok is True
    assert "joao" in _carregar_usuarios()


def test_criar_usuario_duplicado(tmp_users):
    from auth import criar_usuario
    criar_usuario("maria", "abc", "viewer")
    ok, msg = criar_usuario("maria", "xyz", "admin")
    assert ok is False
    assert "já existe" in msg.lower()


def test_criar_usuario_role_invalido(tmp_users):
    from auth import criar_usuario
    ok, msg = criar_usuario("teste", "abc", "superadmin")
    assert ok is False


def test_alterar_senha(tmp_users):
    from auth import criar_usuario, alterar_senha, _verificar_senha, _carregar_usuarios
    criar_usuario("carlos", "senha_antiga", "viewer")
    ok, _ = alterar_senha("carlos", "senha_antiga", "senha_nova")
    assert ok is True
    u = _carregar_usuarios()["carlos"]
    assert _verificar_senha("senha_nova", u["hash"], u["salt"]) is True


def test_alterar_senha_errada(tmp_users):
    from auth import criar_usuario, alterar_senha
    criar_usuario("ana", "correta", "viewer")
    ok, msg = alterar_senha("ana", "errada", "nova")
    assert ok is False
