"""
auth.py — Autenticação, sessão, permissões e rate-limiting.

Streamlit Cloud: usuários lidos de st.secrets["usuarios"] (sem users.json).
Local / servidor próprio: users.json em config/.
"""
from __future__ import annotations
import os, json, time, hashlib, hmac as _hmac, logging
import streamlit as st
import pandas as pd
from pathlib import Path
from config import get_settings

logger     = logging.getLogger("plenodoc.auth")
CONFIG_DIR = Path("config")
USERS_FILE = CONFIG_DIR / "users.json"

PERMISSOES: dict[str, set] = {
    "admin":    {"gerenciar_docs","configurar_modelo","gerenciar_banco","chat","banco_query","ver_audit","ver_metricas"},
    "operator": {"chat","banco_query","ver_metricas"},
    "viewer":   {"chat"},
}

# ── Hashing ───────────────────────────────────────────────────────────────────

def _hash_senha(senha: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(32)
    dk   = hashlib.pbkdf2_hmac("sha256", senha.encode("utf-8"), salt, 260_000)
    return dk.hex(), salt.hex()

def _verificar_senha(senha: str, hash_hex: str, salt_hex: str) -> bool:
    comp, _ = _hash_senha(senha, salt_hex)
    return _hmac.compare_digest(comp.encode(), hash_hex.encode())

def hmac_compare(a: str, b: str) -> bool:
    return _hmac.compare_digest(a.encode(), b.encode())

# ── Fonte de usuários: st.secrets (Cloud) ou users.json (local) ──────────────

def _usuarios_do_secrets() -> dict | None:
    """
    Retorna usuários definidos em st.secrets["usuarios"] ou None.
    Senhas em texto puro são hasheadas na primeira leitura e salvas em cache.
    """
    try:
        raw = st.secrets.get("usuarios", {})
        if not raw:
            return None
        usuarios = {}
        for username, senha_ou_hash in raw.items():
            # Se vier como string simples (texto puro), faz hash agora
            if isinstance(senha_ou_hash, str) and len(senha_ou_hash) != 64:
                h, s = _hash_senha(senha_ou_hash)
                usuarios[username] = {"hash": h, "salt": s, "role": "admin", "display_name": username}
            else:
                # Já está no formato dict com hash/salt
                usuarios[username] = dict(senha_ou_hash)
        return usuarios if usuarios else None
    except Exception:
        return None


def _carregar_usuarios() -> dict:
    # Prioridade 1: st.secrets (Streamlit Cloud)
    u_secrets = _usuarios_do_secrets()
    if u_secrets:
        return u_secrets

    # Prioridade 2: users.json local
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        h, s = _hash_senha("1234")
        _salvar_usuarios({"Administrador": {
            "hash": h, "salt": s, "role": "admin", "display_name": "Administrador"
        }})
        logger.info("users.json criado com admin padrão (senha: 1234).")
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Falha ao ler users.json: %s", e)
        return {}


def _salvar_usuarios(u: dict) -> None:
    """Salva somente no modo local (não aplicável ao Cloud via secrets)."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        USERS_FILE.write_text(json.dumps(u, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning("Não foi possível salvar users.json: %s", e)


# ── CRUD de usuários ──────────────────────────────────────────────────────────

def criar_usuario(username: str, senha: str, role: str = "viewer", display_name: str = "") -> tuple[bool, str]:
    if _usuarios_do_secrets():
        return False, "Usuários gerenciados via st.secrets no Streamlit Cloud. Adicione lá."
    u = _carregar_usuarios()
    if username in u:          return False, f"Usuário '{username}' já existe."
    if role not in PERMISSOES: return False, f"Perfil inválido: {role}."
    h, s = _hash_senha(senha)
    u[username] = {"hash": h, "salt": s, "role": role, "display_name": display_name or username}
    _salvar_usuarios(u)
    from audit import registrar
    registrar("criar_usuario", f"username={username} role={role}")
    return True, f"Usuário '{username}' criado."


def alterar_senha(username: str, senha_atual: str, nova_senha: str) -> tuple[bool, str]:
    if _usuarios_do_secrets():
        return False, "Altere a senha diretamente nos Secrets do Streamlit Cloud."
    u = _carregar_usuarios()
    d = u.get(username)
    if not d or not _verificar_senha(senha_atual, d["hash"], d["salt"]):
        return False, "Senha atual incorreta."
    h, s = _hash_senha(nova_senha)
    u[username]["hash"] = h
    u[username]["salt"] = s
    _salvar_usuarios(u)
    from audit import registrar
    registrar("alterar_senha", f"username={username}")
    return True, "Senha alterada com sucesso."


# ── Rate-limiting ─────────────────────────────────────────────────────────────

def verificar_rate_limit() -> tuple[bool, str]:
    cfg   = get_settings()
    agora = time.time()
    reqs  = [t for t in st.session_state.get("rl_requests", []) if agora - t < 60]
    reqs.append(agora)
    st.session_state.rl_requests = reqs
    if len(reqs) > cfg.max_requests_per_minute:
        espera = int(60 - (agora - reqs[0]))
        return False, f"⏱️ Limite de {cfg.max_requests_per_minute} req/min atingido. Aguarde {espera}s."
    return True, ""


# ── Sessão ────────────────────────────────────────────────────────────────────

def verificar_timeout() -> bool:
    cfg = get_settings()
    if not st.session_state.get("logged_in") or cfg.session_timeout_minutes <= 0:
        return False
    if time.time() - st.session_state.get("last_activity", time.time()) > cfg.session_timeout_minutes * 60:
        from audit import registrar
        registrar("timeout_sessao")
        _limpar_sessao()
        st.warning("⏱️ Sessão encerrada por inatividade.")
        return True
    st.session_state.last_activity = time.time()
    return False


def _limpar_sessao() -> None:
    for k in ["logged_in","username","user_role","display_name","chain","retriever","llm","last_activity"]:
        st.session_state[k] = False if k == "logged_in" else None


def tem_permissao(acao: str) -> bool:
    return acao in PERMISSOES.get(st.session_state.get("user_role", "viewer"), set())


def usuario_atual() -> dict:
    return {
        "username":     st.session_state.get("username", ""),
        "role":         st.session_state.get("user_role", "viewer"),
        "display_name": st.session_state.get("display_name", ""),
    }


# ── Login UI ──────────────────────────────────────────────────────────────────

def pagina_login() -> None:
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
                st.session_state.logged_in     = True
                st.session_state.username      = usuario.strip()
                st.session_state.user_role     = dados.get("role", "viewer")
                st.session_state.display_name  = dados.get("display_name", usuario.strip())
                st.session_state.last_activity = time.time()
                from audit import registrar
                registrar("login", f"role={dados.get('role')}")
                st.rerun()
            else:
                from audit import registrar
                registrar("login_falhou", f"username={usuario.strip()}")
                st.error("❌ Credenciais inválidas.")


# ── Painel admin ──────────────────────────────────────────────────────────────

def painel_usuarios() -> None:
    st.subheader("👥 Gerenciar Usuários")

    if _usuarios_do_secrets():
        st.info("👆 Usuários gerenciados via **Secrets** no painel do Streamlit Cloud.\nAcesse: App → Settings → Secrets")
        st.code("""
# Formato em Secrets (App Settings → Secrets):
[usuarios]
Administrador = "sua_senha"
Operador      = "outra_senha"
        """, language="toml")
        return

    usuarios = _carregar_usuarios()
    rows = [{"Usuário": u, "Perfil": d["role"], "Nome": d.get("display_name", u)} for u, d in usuarios.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()
    with st.expander("➕ Novo Usuário"):
        with st.form("form_new_user"):
            nu, nn = st.columns(2)
            u_nome    = nu.text_input("Usuário")
            u_display = nn.text_input("Nome de exibição")
            u_senha   = st.text_input("Senha", type="password")
            u_role    = st.selectbox("Perfil", list(PERMISSOES.keys()))
            if st.form_submit_button("Criar", type="primary"):
                if u_nome and u_senha:
                    ok, msg = criar_usuario(u_nome.strip(), u_senha, u_role, u_display.strip())
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()
                else:
                    st.warning("Preencha usuário e senha.")
