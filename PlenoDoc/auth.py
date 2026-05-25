"""
auth.py — Autenticação, controle de sessão e permissões por perfil.

Melhorias aplicadas:
  - Senhas com hash PBKDF2-HMAC-SHA256 + salt (260k iterações — recomendação OWASP)
  - Usuários armazenados em config/users.json (nunca em texto puro)
  - Múltiplos perfis: admin | operator | viewer
  - Timeout automático de sessão por inatividade
  - Controle granular de permissões por ação
"""

import os
import json
import time
import hashlib
import logging
import streamlit as st
from pathlib import Path

logger = logging.getLogger("plenodoc.auth")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CONFIG_DIR              = Path("config")
USERS_FILE              = CONFIG_DIR / "users.json"
SESSION_TIMEOUT_MINUTES = 60        # 0 = sem expiração

# Permissões por perfil
PERMISSOES: dict[str, set] = {
    "admin":    {"gerenciar_docs", "configurar_modelo", "gerenciar_banco", "chat", "banco_query"},
    "operator": {"chat", "banco_query"},
    "viewer":   {"chat"},
}


# ---------------------------------------------------------------------------
# Hashing de senha (sem dependências externas)
# ---------------------------------------------------------------------------

def _hash_senha(senha: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Retorna (hash_hex, salt_hex) usando PBKDF2-HMAC-SHA256."""
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(32)
    dk   = hashlib.pbkdf2_hmac("sha256", senha.encode("utf-8"), salt, 260_000)
    return dk.hex(), salt.hex()


def _verificar_senha(senha: str, hash_hex: str, salt_hex: str) -> bool:
    computed, _ = _hash_senha(senha, salt_hex)
    return hmac_compare(computed, hash_hex)


def hmac_compare(a: str, b: str) -> bool:
    """Comparação em tempo constante para evitar timing attacks."""
    import hmac as _hmac
    return _hmac.compare_digest(a.encode(), b.encode())


# ---------------------------------------------------------------------------
# Gerenciamento de usuários (users.json)
# ---------------------------------------------------------------------------

def _carregar_usuarios() -> dict:
    """Carrega users.json. Cria arquivo com admin padrão se não existir."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        h, s = _hash_senha("1234")
        usuarios_padrao = {
            "Administrador": {
                "hash":         h,
                "salt":         s,
                "role":         "admin",
                "display_name": "Administrador",
            }
        }
        USERS_FILE.write_text(json.dumps(usuarios_padrao, indent=2, ensure_ascii=False))
        logger.info("Arquivo users.json criado com usuário padrão.")
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Falha ao carregar users.json: %s", e)
        return {}


def _salvar_usuarios(usuarios: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(usuarios, indent=2, ensure_ascii=False))


def criar_usuario(username: str, senha: str, role: str = "viewer", display_name: str = "") -> tuple[bool, str]:
    """Cria um novo usuário. Retorna (sucesso, mensagem)."""
    usuarios = _carregar_usuarios()
    if username in usuarios:
        return False, f"Usuário '{username}' já existe."
    if role not in PERMISSOES:
        return False, f"Perfil inválido: {role}. Use: {', '.join(PERMISSOES)}."
    h, s = _hash_senha(senha)
    usuarios[username] = {
        "hash":         h,
        "salt":         s,
        "role":         role,
        "display_name": display_name or username,
    }
    _salvar_usuarios(usuarios)
    logger.info("Usuário '%s' criado com perfil '%s'.", username, role)
    return True, f"Usuário '{username}' criado com sucesso."


def alterar_senha(username: str, senha_atual: str, nova_senha: str) -> tuple[bool, str]:
    usuarios = _carregar_usuarios()
    dados = usuarios.get(username)
    if not dados or not _verificar_senha(senha_atual, dados["hash"], dados["salt"]):
        return False, "Senha atual incorreta."
    h, s = _hash_senha(nova_senha)
    usuarios[username]["hash"] = h
    usuarios[username]["salt"] = s
    _salvar_usuarios(usuarios)
    logger.info("Senha de '%s' alterada.", username)
    return True, "Senha alterada com sucesso."


# ---------------------------------------------------------------------------
# Controle de sessão
# ---------------------------------------------------------------------------

def verificar_timeout() -> bool:
    """
    Verifica inatividade. Retorna True (e desloga) se sessão expirou.
    Deve ser chamada no início de main() a cada rerun.
    """
    if not st.session_state.get("logged_in") or SESSION_TIMEOUT_MINUTES <= 0:
        return False
    ultima = st.session_state.get("last_activity", time.time())
    if time.time() - ultima > SESSION_TIMEOUT_MINUTES * 60:
        logger.info("Sessão expirada por inatividade: %s", st.session_state.get("username"))
        _limpar_sessao()
        st.warning("⏱️ Sessão encerrada por inatividade. Faça login novamente.")
        return True
    st.session_state.last_activity = time.time()
    return False


def _limpar_sessao() -> None:
    for k in ["logged_in", "username", "user_role", "display_name",
              "chain", "retriever", "llm", "last_activity"]:
        st.session_state[k] = False if k == "logged_in" else None


# ---------------------------------------------------------------------------
# Permissões
# ---------------------------------------------------------------------------

def tem_permissao(acao: str) -> bool:
    """Verifica se o usuário logado tem permissão para a ação."""
    role = st.session_state.get("user_role", "viewer")
    return acao in PERMISSOES.get(role, set())


def usuario_atual() -> dict:
    return {
        "username":     st.session_state.get("username", ""),
        "role":         st.session_state.get("user_role", "viewer"),
        "display_name": st.session_state.get("display_name", ""),
    }


# ---------------------------------------------------------------------------
# Interface de login
# ---------------------------------------------------------------------------

def pagina_login() -> None:
    """Renderiza a tela de login centralizada."""
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.title("PlenoDoc 📑")
        st.markdown("Autenticação necessária para acessar a base de conhecimento.")
        st.divider()

        with st.form("login_form"):
            usuario = st.text_input("👤 Usuário", placeholder="Seu usuário")
            senha   = st.text_input("🔒 Senha",   type="password", placeholder="Sua senha")
            entrar  = st.form_submit_button("Entrar →", use_container_width=True, type="primary")

        if entrar:
            if not usuario.strip() or not senha:
                st.warning("Preencha usuário e senha.")
                return

            usuarios = _carregar_usuarios()
            dados    = usuarios.get(usuario.strip())

            if dados and _verificar_senha(senha, dados["hash"], dados["salt"]):
                st.session_state.logged_in    = True
                st.session_state.username     = usuario.strip()
                st.session_state.user_role    = dados.get("role", "viewer")
                st.session_state.display_name = dados.get("display_name", usuario.strip())
                st.session_state.last_activity = time.time()
                logger.info("Login: %s (perfil: %s)", usuario.strip(), dados.get("role"))
                st.rerun()
            else:
                logger.warning("Tentativa de login inválida: %s", usuario.strip())
                st.error("❌ Credenciais inválidas.")


# ---------------------------------------------------------------------------
# Painel de gerenciamento de usuários (somente admin)
# ---------------------------------------------------------------------------

def painel_usuarios() -> None:
    """UI para o admin criar/listar usuários. Exibir apenas para admins."""
    st.subheader("👥 Gerenciar Usuários")
    usuarios = _carregar_usuarios()

    # Lista
    rows = [{"Usuário": u, "Perfil": d["role"], "Nome": d.get("display_name", u)}
            for u, d in usuarios.items()]
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()

    # Criar novo
    with st.expander("➕ Novo Usuário"):
        with st.form("form_novo_usuario"):
            nu   = st.text_input("Usuário")
            nn   = st.text_input("Nome de exibição")
            ns   = st.text_input("Senha", type="password")
            nr   = st.selectbox("Perfil", list(PERMISSOES.keys()))
            ok_btn = st.form_submit_button("Criar", type="primary")
        if ok_btn:
            if nu and ns:
                ok, msg = criar_usuario(nu.strip(), ns, nr, nn.strip())
                st.success(msg) if ok else st.error(msg)
            else:
                st.warning("Preencha usuário e senha.")
