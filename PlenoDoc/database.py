"""
database.py — Módulo de banco de dados do PlenoDoc.

Melhorias aplicadas:
  - Validador SQL (bloqueia DROP/TRUNCATE/DELETE sem WHERE/UPDATE sem WHERE/multi-statement)
  - Cache de schema em disco (JSON) — sobrevive a reinicializações
  - Histórico de queries executadas na sessão
  - Paginação de resultados grandes
  - Diagrama ER gerado com Graphviz DOT (st.graphviz_chart)
  - Logging estruturado
"""

from __future__ import annotations

import io, os, re, json, textwrap, logging
import datetime
import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text, inspect as sa_inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("plenodoc.database")

# ============================================================================
# CONFIGURAÇÃO DOS BANCOS SUPORTADOS
# ============================================================================

BANCOS_SUPORTADOS: dict[str, dict] = {
    "MySQL / MariaDB": {"driver": "mysql+pymysql",       "porta_padrao": 3306,  "icone": "🐬", "dialect_hint": "MySQL",              "requer_host": True},
    "PostgreSQL":      {"driver": "postgresql+psycopg2", "porta_padrao": 5432,  "icone": "🐘", "dialect_hint": "PostgreSQL",          "requer_host": True},
    "SQL Server":      {"driver": "mssql+pymssql",       "porta_padrao": 1433,  "icone": "🪟", "dialect_hint": "T-SQL (SQL Server)",  "requer_host": True},
    "SQLite":          {"driver": "sqlite",               "porta_padrao": None,  "icone": "📁", "dialect_hint": "SQLite",              "requer_host": False},
    "Oracle":          {"driver": "oracle+oracledb",      "porta_padrao": 1521,  "icone": "🔴", "dialect_hint": "Oracle SQL",          "requer_host": True},
}

_SAMPLE_ROWS       = 3
_MAX_SCHEMA_CHARS  = 12_000
_MAX_RESULT_LLM    = 50
_ROWS_PER_PAGE     = 100
_MAX_HIST_QUERIES  = 50
_CACHE_DIR         = Path(".pleno_cache")


# ============================================================================
# DATACLASS DE CONFIGURAÇÃO
# ============================================================================

@dataclass
class ConfigConexao:
    nome:    str
    tipo:    str
    host:    str  = "localhost"
    porta:   int  = 3306
    banco:   str  = ""
    usuario: str  = ""
    senha:   str  = ""
    extras:  dict = field(default_factory=dict)


# ============================================================================
# ESTADO DA SESSÃO
# ============================================================================

def inicializar_estado_db() -> None:
    defaults = {
        "db_configs":          {},
        "db_engines":          {},
        "db_ativo":            None,
        "db_schema_map":       {},
        "db_chat_history":     [],
        "db_query_manual":     "",
        "db_resultado_manual": None,
        "db_query_history":    [],   # [{sql, ts, rows, conn}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# VALIDADOR SQL
# ============================================================================

_RE_BLOQUEADO_SEMPRE = re.compile(
    r"^\s*(DROP|TRUNCATE|ALTER\s+TABLE|CREATE\s+TABLE|GRANT|REVOKE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_RE_DELETE_SEM_WHERE = re.compile(
    r"\bDELETE\s+FROM\s+\S+\s*(?:;|$)", re.IGNORECASE
)
_RE_UPDATE_SEM_WHERE = re.compile(
    r"\bUPDATE\s+\S+\s+SET\b(?:(?!\bWHERE\b).)*(?:;|$)", re.IGNORECASE | re.DOTALL
)
_RE_SOMENTE_LEITURA = re.compile(
    r"^\s*(SELECT|WITH|SHOW|EXPLAIN|DESCRIBE|PRAGMA)\b", re.IGNORECASE
)


def _validar_sql(sql: str, modo: str = "manual") -> tuple[bool, list[str]]:
    """
    Valida segurança do SQL antes de executar.

    modo='llm'    — apenas SELECT/WITH/SHOW/EXPLAIN/DESCRIBE permitidos
    modo='manual' — bloqueia comandos destrutivos, avisa sobre operações sem WHERE

    Retorna (é_seguro, lista_de_erros_ou_avisos).
    """
    erros: list[str] = []

    # Multi-statement
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    if len(stmts) > 1:
        if modo == "llm":
            erros.append("❌ Multi-statement bloqueado no modo automático.")
            return False, erros
        else:
            erros.append(f"⚠️ {len(stmts)} comandos detectados — execute um de cada vez.")

    for stmt in stmts:
        # Comandos sempre bloqueados
        if _RE_BLOQUEADO_SEMPRE.match(stmt):
            cmd = stmt.split()[0].upper()
            erros.append(f"❌ Comando '{cmd}' bloqueado por segurança.")
            return False, erros

        if modo == "llm":
            if not _RE_SOMENTE_LEITURA.match(stmt):
                erros.append("❌ Apenas consultas de leitura (SELECT/WITH/SHOW) são permitidas no modo automático.")
                return False, erros
        else:
            # Modo manual — bloqueia DELETE/UPDATE sem WHERE
            if _RE_DELETE_SEM_WHERE.search(stmt):
                erros.append("❌ DELETE sem cláusula WHERE bloqueado. Adicione uma condição WHERE.")
                return False, erros
            if _RE_UPDATE_SEM_WHERE.search(stmt):
                erros.append("❌ UPDATE sem cláusula WHERE bloqueado. Adicione uma condição WHERE.")
                return False, erros

    return len(erros) == 0 or all(e.startswith("⚠️") for e in erros), erros


# ============================================================================
# CONEXÃO
# ============================================================================

def _build_url(cfg: ConfigConexao) -> str:
    import urllib.parse
    if cfg.tipo == "SQLite":
        return f"sqlite:///{cfg.banco.strip() or ':memory:'}"
    porta = f":{cfg.porta}" if cfg.porta else ""
    cred  = ""
    if cfg.usuario:
        cred = f"{urllib.parse.quote_plus(cfg.usuario)}:{urllib.parse.quote_plus(cfg.senha)}@"
    return f"{BANCOS_SUPORTADOS[cfg.tipo]['driver']}://{cred}{cfg.host}{porta}/{cfg.banco}"


def conectar(cfg: ConfigConexao) -> tuple[bool, str]:
    try:
        kwargs = {"pool_pre_ping": True}
        if cfg.tipo != "SQLite":
            kwargs["connect_args"] = {"connect_timeout": 10}
        engine = create_engine(_build_url(cfg), **kwargs)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.session_state.db_configs[cfg.nome] = cfg
        st.session_state.db_engines[cfg.nome] = engine
        st.session_state.db_ativo             = cfg.nome
        logger.info("Conectado: %s (%s)", cfg.nome, cfg.tipo)
        return True, f"Conectado a **{cfg.nome}** ({cfg.tipo})"
    except SQLAlchemyError as e:
        return False, f"Erro de conexão: {_limpar_erro(str(e))}"
    except Exception as e:
        return False, f"Erro inesperado: {e}"


def desconectar(nome: str) -> None:
    eng = st.session_state.db_engines.pop(nome, None)
    if eng:
        eng.dispose()
    st.session_state.db_configs.pop(nome, None)
    st.session_state.db_schema_map.pop(nome, None)
    if st.session_state.db_ativo == nome:
        restantes = list(st.session_state.db_engines.keys())
        st.session_state.db_ativo = restantes[0] if restantes else None


def engine_ativo():
    nome = st.session_state.get("db_ativo")
    return st.session_state.db_engines.get(nome) if nome else None


# ============================================================================
# MAPEAMENTO DE SCHEMA
# ============================================================================

def mapear_banco(nome_conexao: str) -> tuple[bool, str]:
    engine = st.session_state.db_engines.get(nome_conexao)
    if not engine:
        return False, "Engine não encontrado."

    cfg  = st.session_state.db_configs[nome_conexao]
    insp = sa_inspect(engine)
    schema_map = {"tipo": cfg.tipo, "banco": cfg.banco, "tabelas": {}}

    try:
        tabelas = sorted(insp.get_table_names())
    except Exception as e:
        return False, f"Erro ao listar tabelas: {e}"

    prog  = st.progress(0, text="Mapeando schema…")
    total = max(len(tabelas), 1)

    for i, tabela in enumerate(tabelas):
        prog.progress((i + 1) / total, text=f"Mapeando {tabela}…")
        info = {"colunas": [], "pks": [], "fks": [], "indices": [], "row_count": None, "sample_data": []}

        try:
            info["colunas"] = [
                {"nome": c["name"], "tipo": str(c["type"]),
                 "nullable": c.get("nullable", True), "default": str(c.get("default") or "")}
                for c in insp.get_columns(tabela)
            ]
        except Exception: pass

        try: info["pks"] = insp.get_pk_constraint(tabela).get("constrained_columns", [])
        except Exception: pass

        try:
            info["fks"] = [
                {"colunas_locais": fk.get("constrained_columns", []),
                 "tabela_ref": fk.get("referred_table", ""),
                 "colunas_ref": fk.get("referred_columns", [])}
                for fk in insp.get_foreign_keys(tabela)
            ]
        except Exception: pass

        try:
            info["indices"] = [
                {"nome": idx.get("name", ""), "colunas": idx.get("column_names", []), "unique": idx.get("unique", False)}
                for idx in insp.get_indexes(tabela)
            ]
        except Exception: pass

        try:
            with engine.connect() as conn:
                info["row_count"] = conn.execute(text(f"SELECT COUNT(*) FROM {tabela}")).scalar()
        except Exception:
            info["row_count"] = -1

        try:
            with engine.connect() as conn:
                res  = conn.execute(text(f"SELECT * FROM {tabela} LIMIT {_SAMPLE_ROWS}"))
                cols = list(res.keys())
                info["sample_data"] = [dict(zip(cols, [str(v) for v in row])) for row in res.fetchall()]
        except Exception: pass

        schema_map["tabelas"][tabela] = info

    prog.empty()
    st.session_state.db_schema_map[nome_conexao] = schema_map
    salvar_cache_schema(nome_conexao)
    logger.info("Schema mapeado: %s (%d tabelas)", nome_conexao, len(tabelas))
    return True, f"Schema mapeado: {len(tabelas)} tabela(s)."


# ============================================================================
# CACHE DE SCHEMA EM DISCO
# ============================================================================

def _cache_path(nome: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"schema_{re.sub(r'[^a-zA-Z0-9_-]', '_', nome)}.json"


def salvar_cache_schema(nome: str) -> None:
    schema = st.session_state.db_schema_map.get(nome)
    if schema:
        _cache_path(nome).write_text(json.dumps(schema, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("Cache de schema salvo: %s", nome)


def carregar_cache_schema(nome: str) -> bool:
    path = _cache_path(nome)
    if not path.exists():
        return False
    try:
        st.session_state.db_schema_map[nome] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Cache de schema carregado do disco: %s", nome)
        return True
    except Exception as e:
        logger.warning("Falha ao carregar cache de schema: %s", e)
        return False


# ============================================================================
# CONTEXTO TEXTUAL DO SCHEMA
# ============================================================================

def gerar_contexto_schema(nome_conexao: str) -> str:
    schema_map = st.session_state.db_schema_map.get(nome_conexao)
    if not schema_map:
        return "(schema não mapeado)"
    cfg    = st.session_state.db_configs.get(nome_conexao)
    linhas = [f"=== SCHEMA: {schema_map['tipo']} | {schema_map['banco'] or nome_conexao} ===", ""]

    for tabela, info in schema_map["tabelas"].items():
        count = info.get("row_count", "?")
        count_str = f"{count:,}" if isinstance(count, int) and count >= 0 else "?"
        linhas.append(f"TABELA: {tabela}  [{count_str} linhas]")
        for col in info.get("colunas", []):
            flags = ["PK"] if col["nome"] in info.get("pks", []) else []
            for fk in info.get("fks", []):
                if col["nome"] in fk.get("colunas_locais", []):
                    flags.append(f"FK→{fk['tabela_ref']}")
            nulo  = "NULL" if col["nullable"] else "NOT NULL"
            linhas.append(f"  {col['nome']:<25} {col['tipo']:<18} {nulo}  {', '.join(flags)}")
        for fk in info.get("fks", []):
            linhas.append(f"  FK: {tabela}.{', '.join(fk['colunas_locais'])} → {fk['tabela_ref']}.{', '.join(fk['colunas_ref'])}")
        if info.get("sample_data"):
            linhas.append(f"  Amostra:")
            if info["sample_data"]:
                linhas.append("    " + " | ".join(f"{k[:12]}" for k in info["sample_data"][0].keys()))
                for row in info["sample_data"]:
                    linhas.append("    " + " | ".join(f"{str(v)[:12]}" for v in row.values()))
        linhas.append("")

    texto = "\n".join(linhas)
    if len(texto) > _MAX_SCHEMA_CHARS:
        texto = texto[:_MAX_SCHEMA_CHARS] + f"\n[... schema truncado — {len(schema_map['tabelas'])} tabelas no total ...]"
    return texto


# ============================================================================
# DIAGRAMA ER (Graphviz DOT)
# ============================================================================

def gerar_diagrama_er(nome_conexao: str) -> str:
    """Gera string DOT para st.graphviz_chart."""
    schema_map = st.session_state.db_schema_map.get(nome_conexao, {})
    tabelas    = schema_map.get("tabelas", {})

    dot = [
        "digraph ER {",
        "  rankdir=LR;",
        "  graph [fontsize=10 fontname=Helvetica];",
        '  node [shape=record fontsize=9 style="filled,rounded" fillcolor="#FFF9C4" color="#888"];',
        "  edge [fontsize=8 color=navy arrowhead=crow arrowtail=none dir=both];",
        "",
    ]

    for tabela, info in tabelas.items():
        safe = re.sub(r"\W", "_", tabela)
        count = info.get("row_count", -1)
        count_str = f"{count:,}" if isinstance(count, int) and count >= 0 else "?"
        # Escape special chars from values
        def esc(s): return str(s).replace('"', "'").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("|", "∣")
        cols_str = ""
        for col in info.get("colunas", [])[:14]:
            flags = []
            if col["nome"] in info.get("pks", []):      flags.append("PK")
            for fk in info.get("fks", []):
                if col["nome"] in fk.get("colunas_locais", []): flags.append("FK")
            tipo   = esc(str(col["tipo"]).split("(")[0][:8])
            flag_s = f"[{','.join(flags)}]" if flags else ""
            null_s = "" if col.get("nullable") else "✱"
            cols_str += f"{esc(col['nome'])}{null_s}: {tipo} {flag_s}\\l"
        extra = ""
        if len(info.get("colunas", [])) > 14:
            extra = f"... +{len(info['colunas'])-14} cols\\l"
        label = "{" + f"{esc(tabela)} ({count_str})" + "|" + cols_str + extra + "}"
        dot.append(f'  {safe} [label="{label}"];')

    dot.append("")

    for tabela, info in tabelas.items():
        safe_from = re.sub(r"\W", "_", tabela)
        for fk in info.get("fks", []):
            safe_to = re.sub(r"\W", "_", fk.get("tabela_ref", ""))
            col_lbl = fk.get("colunas_locais", [""])[0]
            if safe_to in [re.sub(r"\W", "_", t) for t in tabelas]:
                dot.append(f'  {safe_from} -> {safe_to} [label="{col_lbl}"];')

    dot.append("}")
    return "\n".join(dot)


# ============================================================================
# HISTÓRICO DE QUERIES
# ============================================================================

def _registrar_historico(sql: str, rows: int | None, conexao: str) -> None:
    hist = st.session_state.setdefault("db_query_history", [])
    hist.insert(0, {
        "sql":   sql,
        "ts":    datetime.datetime.now().strftime("%H:%M:%S"),
        "rows":  rows if rows is not None else "-",
        "conn":  conexao,
    })
    st.session_state.db_query_history = hist[:_MAX_HIST_QUERIES]


# ============================================================================
# EXECUÇÃO DE QUERIES
# ============================================================================

def executar_query(sql: str, nome_conexao: str | None = None, modo_validacao: str = "manual") -> tuple[bool, str, pd.DataFrame | None]:
    nome   = nome_conexao or st.session_state.get("db_ativo")
    engine = st.session_state.db_engines.get(nome) if nome else None
    if not engine:
        return False, "Nenhuma conexão ativa.", None

    sql = sql.strip()
    if not sql:
        return False, "Query vazia.", None

    # Validação de segurança
    seguro, msgs = _validar_sql(sql, modo=modo_validacao)
    if not seguro:
        return False, "\n".join(msgs), None
    for aviso in [m for m in msgs if m.startswith("⚠️")]:
        st.warning(aviso)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
                _registrar_historico(sql, len(df), nome)
                return True, f"✅ {len(df):,} linha(s) retornada(s).", df
            conn.commit()
            n = result.rowcount
            _registrar_historico(sql, n, nome)
            return True, f"✅ Executado — {n if n >= 0 else '?'} linha(s) afetada(s).", None

    except SQLAlchemyError as e:
        return False, f"❌ Erro SQL: {_limpar_erro(str(e))}", None
    except Exception as e:
        return False, f"❌ {e}", None


# ============================================================================
# PIPELINE NL-to-SQL
# ============================================================================

def _extrair_sql(texto: str) -> str:
    texto = re.sub(r"```(?:sql)?\s*", "", texto, flags=re.IGNORECASE)
    texto = texto.replace("```", "").strip()
    return "\n".join(l for l in texto.splitlines() if l.strip())


def _df_para_texto(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(nenhum resultado)"
    truncado = len(df) > _MAX_RESULT_LLM
    texto    = df.head(_MAX_RESULT_LLM).to_string(index=False, max_colwidth=40)
    return texto + (f"\n[... {len(df) - _MAX_RESULT_LLM} linha(s) omitida(s) ...]" if truncado else "")


def responder_pergunta_banco(pergunta: str, nome_conexao: str) -> dict:
    llm = st.session_state.get("llm")
    if not llm:
        return {"erro": "Modelo LLM não inicializado. Configure na aba 🤖 Modelo."}

    schema_texto = gerar_contexto_schema(nome_conexao)
    if "(schema não mapeado)" in schema_texto:
        return {"erro": "Schema não mapeado. Clique em 🗺️ Mapear Banco primeiro."}

    cfg     = st.session_state.db_configs.get(nome_conexao)
    dialeto = BANCOS_SUPORTADOS[cfg.tipo]["dialect_hint"] if cfg else "SQL"

    # Passo 1 — geração do SQL
    try:
        resp_sql = llm.invoke([
            SystemMessage(content=textwrap.dedent(f"""\
                Você é especialista em {dialeto}.
                Com base no schema abaixo, gere SOMENTE o SQL que responde à pergunta.
                Regras: retorne APENAS SQL puro sem markdown, use só tabelas/colunas do schema,
                adicione LIMIT 200 em consultas amplas, nunca gere DROP/DELETE/UPDATE/INSERT/TRUNCATE.
                {schema_texto}""")),
            HumanMessage(content=pergunta),
        ])
        sql_gerado = _extrair_sql(resp_sql.content)
    except Exception as e:
        return {"erro": f"Falha ao gerar SQL: {e}"}

    if not sql_gerado:
        return {"erro": "LLM não retornou SQL válido."}

    # Passo 2 — execução (modo llm = só leitura)
    ok, msg, df = executar_query(sql_gerado, nome_conexao, modo_validacao="llm")

    if not ok:
        return {"sql": sql_gerado, "ok": False, "msg": msg, "df": None, "resposta": None, "erro": msg}

    # Passo 3 — interpretação do resultado
    resultado_texto = _df_para_texto(df) if df is not None else "(comando executado sem retorno)"

    try:
        resp_final = llm.invoke([
            SystemMessage(content="Você é um analista de dados. Com base na pergunta, no SQL executado e nos resultados, forneça uma resposta clara e objetiva em português. Se vazio, informe claramente. Não explique o SQL."),
            HumanMessage(content=f"Pergunta: {pergunta}\n\nSQL:\n{sql_gerado}\n\nResultado:\n{resultado_texto}"),
        ])
        resposta = resp_final.content
    except Exception as e:
        resposta = f"(Erro ao interpretar resultado: {e})"

    return {"sql": sql_gerado, "ok": True, "msg": msg, "df": df, "resposta": resposta, "erro": None}


# ============================================================================
# PAGINAÇÃO
# ============================================================================

def _paginador(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if len(df) <= _ROWS_PER_PAGE:
        return df
    total_pages = (len(df) + _ROWS_PER_PAGE - 1) // _ROWS_PER_PAGE
    c1, c2 = st.columns([0.3, 0.7])
    page   = c1.number_input(f"Página (1–{total_pages})", 1, total_pages, 1, key=f"pg_{key}")
    c2.caption(f"Exibindo {_ROWS_PER_PAGE} de **{len(df):,}** linhas")
    start = (page - 1) * _ROWS_PER_PAGE
    return df.iloc[start : start + _ROWS_PER_PAGE]


# ============================================================================
# EXPORTAR PARA RAG
# ============================================================================

def exportar_para_rag(df: pd.DataFrame, nome: str = "query") -> tuple[bool, str]:
    from data_processing import adicionar_ao_indice
    try:
        os.makedirs("dados_docs", exist_ok=True)
        caminho = os.path.join("dados_docs", f"db_{nome}.csv")
        df.to_csv(caminho, index=False, encoding="utf-8")
        return adicionar_ao_indice([caminho])
    except Exception as e:
        return False, f"Erro ao exportar para RAG: {e}"


# ============================================================================
# HELPERS
# ============================================================================

def _limpar_erro(msg: str) -> str:
    return " ".join(l for l in msg.splitlines()[:3] if not l.strip().startswith("(Background"))


def _botoes_download(df: pd.DataFrame, prefix: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ CSV", df.to_csv(index=False).encode(), "resultado.csv", "text/csv", use_container_width=True, key=f"{prefix}_csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False)
    c2.download_button("⬇️ Excel", buf.getvalue(), "resultado.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key=f"{prefix}_xl")
    if c3.button("📚 → RAG", use_container_width=True, key=f"{prefix}_rag", help="Indexa resultado na base RAG"):
        ok, m = exportar_para_rag(df, prefix)
        st.success(m) if ok else st.error(m)


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def painel_conexao_db() -> None:
    inicializar_estado_db()
    conexoes = st.session_state.db_configs

    if conexoes:
        st.subheader("🔌 Conexões")
        for nome, cfg in list(conexoes.items()):
            icone  = BANCOS_SUPORTADOS[cfg.tipo]["icone"]
            ativo  = st.session_state.db_ativo == nome
            mapeado = nome in st.session_state.db_schema_map
            c1, c2, c3 = st.columns([0.55, 0.25, 0.20])
            c1.markdown(f"**{icone} {nome}**{'🗺️' if mapeado else ''}{' ●' if ativo else ''}")
            if not ativo:
                if c2.button("Usar", key=f"usar_{nome}", use_container_width=True):
                    st.session_state.db_ativo = nome
                    carregar_cache_schema(nome)
                    st.rerun()
            else:
                c2.success("Ativa", icon="✅")
            if c3.button("✕", key=f"desc_{nome}", use_container_width=True):
                desconectar(nome); st.rerun()
        st.divider()

    with st.expander("➕ Nova Conexão", expanded=not bool(conexoes)):
        tipo   = st.selectbox("Banco", list(BANCOS_SUPORTADOS.keys()), format_func=lambda t: f"{BANCOS_SUPORTADOS[t]['icone']}  {t}", key="db_form_tipo")
        info   = BANCOS_SUPORTADOS[tipo]
        nome_c = st.text_input("Nome", placeholder="ex: producao", key="db_form_nome")

        if not info["requer_host"]:
            banco  = st.text_input("Arquivo .db", placeholder="caminho/arquivo.db", key="db_sqlite")
            host, porta, usuario, senha = "", None, "", ""
        else:
            ch, cp = st.columns([0.7, 0.3])
            host   = ch.text_input("Host",  value="localhost", key="db_form_host")
            porta  = cp.number_input("Porta", value=info["porta_padrao"], min_value=1, max_value=65535, key="db_form_porta")
            banco  = st.text_input("Banco", placeholder=info.get("placeholder_banco",""), key="db_form_banco")
            usuario = st.text_input("Usuário", key="db_form_usuario")
            senha   = st.text_input("Senha",   type="password", key="db_form_senha")

        if st.button("🔗 Conectar", type="primary", use_container_width=True):
            if not nome_c.strip():
                st.error("Defina um nome.")
            elif nome_c in st.session_state.db_configs:
                st.error(f"Já existe conexão '{nome_c}'.")
            else:
                cfg = ConfigConexao(nome=nome_c.strip(), tipo=tipo, host=host, porta=int(porta or 0), banco=banco, usuario=usuario, senha=senha)
                with st.spinner("Testando conexão…"):
                    ok, msg = conectar(cfg)
                if ok:
                    carregar_cache_schema(nome_c.strip())
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)


def pagina_banco_dados() -> None:
    inicializar_estado_db()
    st.title("🗄️ Banco de Dados")
    st.caption("Mapeie o schema, converse em linguagem natural, execute SQL e visualize relacionamentos.")

    conexoes = st.session_state.db_configs
    if not conexoes:
        st.info("Nenhuma conexão configurada. Use a aba **🗄️ Banco** na barra lateral.")
        return

    # Seletor de conexão
    nomes     = list(conexoes.keys())
    idx_atual = nomes.index(st.session_state.db_ativo) if st.session_state.db_ativo in nomes else 0
    sel = st.selectbox("Conexão ativa", nomes, index=idx_atual,
                       format_func=lambda n: f"{BANCOS_SUPORTADOS[conexoes[n].tipo]['icone']}  {n}  ({conexoes[n].tipo})")
    if sel != st.session_state.db_ativo:
        st.session_state.db_ativo = sel
        st.session_state.db_resultado_manual = None
        carregar_cache_schema(sel)
        st.rerun()

    cfg_ativa = conexoes[st.session_state.db_ativo]
    mapeado   = st.session_state.db_ativo in st.session_state.db_schema_map

    c_info, c_map, c_status = st.columns([0.55, 0.25, 0.20])
    c_info.caption(f"**Host:** {cfg_ativa.host or 'local'}  |  **Banco:** {cfg_ativa.banco or '—'}  |  **User:** {cfg_ativa.usuario or '—'}")
    if c_map.button("🗺️ Mapear Banco", use_container_width=True, type="primary" if not mapeado else "secondary"):
        with st.spinner("Mapeando schema completo…"):
            ok, msg = mapear_banco(st.session_state.db_ativo)
        st.success(msg) if ok else st.error(msg)
        if ok: st.rerun()

    if mapeado:
        n = len(st.session_state.db_schema_map[st.session_state.db_ativo]["tabelas"])
        c_status.success(f"✅ {n} tabelas")
    else:
        c_status.warning("⚠️ Não mapeado")

    st.divider()

    tab_chat, tab_schema, tab_er, tab_sql, tab_hist = st.tabs([
        "💬 Chat NL", "🗺️ Schema", "🔗 Diagrama ER", "✏️ SQL Manual", "📋 Histórico"
    ])

    # ── Chat NL-to-SQL ──────────────────────────────────────────────────────
    with tab_chat:
        if not mapeado:
            st.info("👆 Mapeie o banco primeiro para usar o chat em linguagem natural.")
        elif not st.session_state.get("llm"):
            st.warning("⚠️ LLM não inicializado. Configure na aba **🤖 Modelo** da barra lateral.")
        else:
            st.caption("Faça perguntas em português. O PlenoDoc gera SQL, executa e responde automaticamente.")
            for entry in st.session_state.db_chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
                    if entry.get("sql"):
                        with st.expander("🔍 SQL gerado", expanded=False):
                            st.code(entry["sql"], language="sql")
                    if entry.get("df") is not None and not entry["df"].empty:
                        with st.expander(f"📊 Dados ({len(entry['df']):,} linhas)", expanded=False):
                            st.dataframe(_paginador(entry["df"], f"hist_{id(entry)}"), use_container_width=True, hide_index=True)

            if pergunta := st.chat_input("Ex: Quantos clientes cadastrados este mês?"):
                with st.chat_message("user"):
                    st.markdown(pergunta)
                st.session_state.db_chat_history.append({"role": "user", "content": pergunta, "sql": None, "df": None})

                with st.chat_message("assistant"):
                    with st.spinner("Gerando SQL e consultando o banco…"):
                        res = responder_pergunta_banco(pergunta, st.session_state.db_ativo)

                    if res.get("erro") and not res.get("sql"):
                        st.error(res["erro"])
                        st.session_state.db_chat_history.append({"role": "assistant", "content": res["erro"], "sql": None, "df": None})
                    else:
                        resposta_md = res.get("resposta") or res.get("erro") or "Sem resposta."
                        st.markdown(resposta_md)
                        if res.get("sql"):
                            with st.expander("🔍 SQL gerado", expanded=False):
                                st.code(res["sql"], language="sql")
                        df = res.get("df")
                        if df is not None and not df.empty:
                            with st.expander(f"📊 {len(df):,} linha(s)", expanded=True):
                                st.dataframe(_paginador(df, f"chat_{len(st.session_state.db_chat_history)}"), use_container_width=True, hide_index=True)
                                _botoes_download(df, f"nl_{len(st.session_state.db_chat_history)}")
                        st.session_state.db_chat_history.append({"role": "assistant", "content": resposta_md, "sql": res.get("sql"), "df": df})

            if st.session_state.db_chat_history:
                if st.button("🗑️ Limpar histórico", use_container_width=True):
                    st.session_state.db_chat_history = []; st.rerun()

    # ── Explorador de Schema ─────────────────────────────────────────────────
    with tab_schema:
        if not mapeado:
            st.info("Mapeie o banco para explorar o schema.")
        else:
            schema_map = st.session_state.db_schema_map[st.session_state.db_ativo]
            tabelas    = schema_map["tabelas"]
            busca = st.text_input("🔍 Filtrar tabela", placeholder="Nome ou parte…")
            tabelas_f = {t: v for t, v in tabelas.items() if busca.lower() in t.lower()} if busca else tabelas
            st.caption(f"{len(tabelas_f)} de {len(tabelas)} tabela(s)")

            for tabela, info in tabelas_f.items():
                count = info.get("row_count", -1)
                count_s = f"{count:,}" if isinstance(count, int) and count >= 0 else "?"
                with st.expander(f"**{tabela}** — {count_s} linhas", expanded=False):
                    cols_df = [{"Coluna": c["nome"], "Tipo": c["tipo"],
                                "Nulo": "✓" if c["nullable"] else "✗",
                                "Flags": " ".join(["🔑 PK"] if c["nome"] in info.get("pks",[]) else []) +
                                         " ".join(f"🔗 FK→{fk['tabela_ref']}" for fk in info.get("fks",[]) if c["nome"] in fk.get("colunas_locais",[]))}
                               for c in info.get("colunas", [])]
                    st.dataframe(pd.DataFrame(cols_df), use_container_width=True, hide_index=True)
                    if info.get("fks"):
                        for fk in info["fks"]:
                            st.markdown(f"🔗 `{tabela}.{', '.join(fk['colunas_locais'])}` → `{fk['tabela_ref']}.{', '.join(fk['colunas_ref'])}`")
                    if info.get("sample_data"):
                        with st.expander("Amostra de dados", expanded=False):
                            st.dataframe(pd.DataFrame(info["sample_data"]), use_container_width=True, hide_index=True)
                    if st.button(f"▶ SELECT * FROM {tabela} LIMIT 100", key=f"sel_{tabela}", use_container_width=True):
                        st.session_state.db_query_manual = f"SELECT *\nFROM   {tabela}\nLIMIT  100;"
                        st.rerun()

    # ── Diagrama ER ──────────────────────────────────────────────────────────
    with tab_er:
        if not mapeado:
            st.info("Mapeie o banco para gerar o diagrama.")
        else:
            tabelas = st.session_state.db_schema_map[st.session_state.db_ativo]["tabelas"]
            if len(tabelas) > 60:
                st.warning(f"⚠️ {len(tabelas)} tabelas detectadas. O diagrama pode ficar grande — considere filtrar.")
            st.caption("Diagrama gerado automaticamente a partir do schema mapeado. FKs são exibidas como arestas.")
            dot = gerar_diagrama_er(st.session_state.db_ativo)
            st.graphviz_chart(dot, use_container_width=True)
            st.download_button("⬇️ Baixar DOT", dot.encode(), "diagrama_er.dot", "text/plain", use_container_width=True)

    # ── Editor SQL Manual ────────────────────────────────────────────────────
    with tab_sql:
        query = st.text_area("SQL", value=st.session_state.db_query_manual, height=200,
                             placeholder="SELECT * FROM tabela LIMIT 100;", label_visibility="collapsed", key="sql_editor")
        st.session_state.db_query_manual = query

        c1, c2 = st.columns([0.2, 0.8])
        run   = c1.button("▶ Executar", type="primary", use_container_width=True)
        clear = c2.button("🗑 Limpar",                   use_container_width=True)

        if clear:
            st.session_state.db_query_manual = ""; st.session_state.db_resultado_manual = None; st.rerun()

        if run:
            if not query.strip():
                st.warning("Digite um SQL antes de executar.")
            else:
                with st.spinner("Executando…"):
                    ok, msg, df = executar_query(query)
                st.session_state.db_resultado_manual = df if ok else None
                st.success(msg) if ok else st.error(msg)

        df_man = st.session_state.get("db_resultado_manual")
        if df_man is not None:
            st.subheader(f"📊 {len(df_man):,} × {len(df_man.columns)} colunas")
            st.dataframe(_paginador(df_man, "manual"), use_container_width=True, hide_index=True)
            _botoes_download(df_man, "man")

    # ── Histórico de Queries ─────────────────────────────────────────────────
    with tab_hist:
        hist = st.session_state.get("db_query_history", [])
        if not hist:
            st.info("Nenhuma query executada nesta sessão.")
        else:
            st.caption(f"Últimas {len(hist)} queries executadas (máx {_MAX_HIST_QUERIES})")
            for i, entry in enumerate(hist):
                rows_s = f"{entry['rows']:,}" if isinstance(entry['rows'], int) and entry['rows'] >= 0 else str(entry['rows'])
                label  = f"🕐 {entry['ts']}  |  {entry['conn']}  |  {rows_s} linhas"
                with st.expander(label, expanded=False):
                    st.code(entry["sql"], language="sql")
                    if st.button("▶ Re-executar", key=f"reexec_{i}", use_container_width=True):
                        st.session_state.db_query_manual = entry["sql"]
                        st.rerun()
            if st.button("🗑️ Limpar histórico", use_container_width=True):
                st.session_state.db_query_history = []; st.rerun()
