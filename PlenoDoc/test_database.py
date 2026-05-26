"""
test_database.py — Testes para validação SQL, Fernet e schema mapping.
"""
import pytest


# ── Validador SQL ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    "SELECT * FROM clientes",
    "SELECT id, nome FROM pedidos WHERE status = 'ativo'",
    "WITH cte AS (SELECT * FROM logs) SELECT * FROM cte",
    "SHOW TABLES",
    "EXPLAIN SELECT * FROM usuarios",
    "DESCRIBE tabela",
])
def test_sql_valido_modo_manual(sql):
    from database import _validar_sql
    ok, msgs = _validar_sql(sql, modo="manual")
    assert ok is True, f"SQL válido rejeitado: {sql!r} | msgs={msgs}"


@pytest.mark.parametrize("sql", [
    "SELECT * FROM clientes",
    "WITH cte AS (SELECT 1) SELECT * FROM cte",
    "SHOW DATABASES",
    "EXPLAIN SELECT 1",
])
def test_sql_valido_modo_llm(sql):
    from database import _validar_sql
    ok, _ = _validar_sql(sql, modo="llm")
    assert ok is True


@pytest.mark.parametrize("sql,modo", [
    ("DROP TABLE clientes",            "manual"),
    ("TRUNCATE TABLE pedidos",         "manual"),
    ("ALTER TABLE users ADD col INT",  "manual"),
    ("DROP TABLE clientes",            "llm"),
    ("INSERT INTO t VALUES (1)",       "llm"),
    ("UPDATE t SET x=1",               "llm"),
    ("DELETE FROM t",                  "llm"),
])
def test_sql_bloqueado(sql, modo):
    from database import _validar_sql
    ok, msgs = _validar_sql(sql, modo=modo)
    assert ok is False, f"SQL perigoso não bloqueado: {sql!r}"
    assert any("❌" in m for m in msgs)


def test_delete_sem_where_bloqueado():
    from database import _validar_sql
    ok, msgs = _validar_sql("DELETE FROM tabela", modo="manual")
    assert ok is False
    assert any("WHERE" in m for m in msgs)


def test_delete_com_where_permitido():
    from database import _validar_sql
    ok, _ = _validar_sql("DELETE FROM tabela WHERE id = 1", modo="manual")
    assert ok is True


def test_update_sem_where_bloqueado():
    from database import _validar_sql
    ok, msgs = _validar_sql("UPDATE tabela SET campo = 'valor'", modo="manual")
    assert ok is False


def test_update_com_where_permitido():
    from database import _validar_sql
    ok, _ = _validar_sql("UPDATE t SET x = 1 WHERE id = 5", modo="manual")
    assert ok is True


def test_multi_statement_bloqueado_modo_llm():
    from database import _validar_sql
    ok, msgs = _validar_sql("SELECT 1; SELECT 2", modo="llm")
    assert ok is False
    assert any("Multi-statement" in m for m in msgs)


def test_multi_statement_aviso_modo_manual():
    from database import _validar_sql
    ok, msgs = _validar_sql("SELECT 1; SELECT 2", modo="manual")
    # Modo manual: dois SELECTs devem passar (apenas aviso)
    assert ok is True


# ── Fernet — Criptografia ─────────────────────────────────────────────────────

def test_cifrar_decifrar(tmp_cache):
    from database import _cifrar, _decifrar
    original = "senha_super_secreta_123!"
    cifrado  = _cifrar(original)
    assert cifrado != original
    assert _decifrar(cifrado) == original


def test_cifrar_string_vazia(tmp_cache):
    from database import _cifrar, _decifrar
    assert _decifrar(_cifrar("")) == ""


def test_fernet_chave_persistida(tmp_cache):
    """A mesma chave deve ser usada entre chamadas (lida do arquivo)."""
    from database import _cifrar, _decifrar
    cifrado = _cifrar("texto")
    # Segunda chamada usa a mesma chave do arquivo
    assert _decifrar(cifrado) == "texto"


# ── Extração de SQL da resposta do LLM ───────────────────────────────────────

@pytest.mark.parametrize("entrada,esperado", [
    ("SELECT * FROM t",                        "SELECT * FROM t"),
    ("```sql\nSELECT 1\n```",                  "SELECT 1"),
    ("```\nSELECT id FROM users\n```",          "SELECT id FROM users"),
    ("Aqui está o SQL:\n```sql\nSELECT 1```",   "Aqui está o SQL:\nSELECT 1"),
])
def test_extrair_sql(entrada, esperado):
    from database import _extrair_sql
    assert _extrair_sql(entrada).strip() == esperado.strip()


# ── Auditoria ─────────────────────────────────────────────────────────────────

def test_registrar_e_carregar_audit(tmp_cache, streamlit_session_state):
    from audit import registrar, carregar_log
    streamlit_session_state["username"]  = "admin_teste"
    streamlit_session_state["user_role"] = "admin"
    registrar("test_acao", "detalhes do teste")
    log = carregar_log()
    assert len(log) >= 1
    ultimo = log[0]
    assert ultimo["usuário"] == "admin_teste"
    assert ultimo["ação"]    == "test_acao"
    assert "detalhes" in ultimo["detalhes"]


def test_audit_sem_usuario_logado(tmp_cache, streamlit_session_state):
    from audit import registrar, carregar_log
    streamlit_session_state.clear()
    registrar("acao_sem_usuario")
    log = carregar_log()
    assert any(r["usuário"] == "sistema" for r in log)
