"""
database.py — Módulo de banco de dados do PlenoDoc.

Melhorias:
  - Fernet: criptografia de senhas em disco
  - Salvar/carregar conexões entre sessões
  - Views e stored procedures no mapeamento
  - SSH tunnel para redes privadas
  - EXPLAIN / plano de execução no editor SQL
  - Validação SQL, cache de schema, histórico, paginação, diagrama ER
"""
from __future__ import annotations

import io, os, re, json, textwrap, logging, datetime
import streamlit as st
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text, inspect as sa_inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("plenodoc.database")

BANCOS_SUPORTADOS: dict[str, dict] = {
    "MySQL / MariaDB": {"driver":"mysql+pymysql",       "porta_padrao":3306, "icone":"🐬","dialect_hint":"MySQL",             "requer_host":True},
    "PostgreSQL":      {"driver":"postgresql+psycopg2", "porta_padrao":5432, "icone":"🐘","dialect_hint":"PostgreSQL",         "requer_host":True},
    "SQL Server":      {"driver":"mssql+pymssql",       "porta_padrao":1433, "icone":"🪟","dialect_hint":"T-SQL (SQL Server)", "requer_host":True},
    "SQLite":          {"driver":"sqlite",               "porta_padrao":None, "icone":"📁","dialect_hint":"SQLite",             "requer_host":False},
    "Oracle":          {"driver":"oracle+oracledb",      "porta_padrao":1521, "icone":"🔴","dialect_hint":"Oracle SQL",         "requer_host":True},
}

_CACHE_DIR        = Path(".pleno_cache")
_CONN_FILE        = _CACHE_DIR / "connections.json"
_KEY_FILE         = _CACHE_DIR / "db.key"
_SAMPLE_ROWS      = 3
_MAX_SCHEMA_CHARS = 12_000
_MAX_RESULT_LLM   = 50
_ROWS_PER_PAGE    = 100
_MAX_HIST         = 50


# ============================================================================
# FERNET — CRIPTOGRAFIA DE SENHAS
# ============================================================================

def _get_fernet():
    from cryptography.fernet import Fernet
    _CACHE_DIR.mkdir(exist_ok=True)
    if not _KEY_FILE.exists():
        _KEY_FILE.write_bytes(Fernet.generate_key())
        logger.info("Chave Fernet gerada.")
    return Fernet(_KEY_FILE.read_bytes())

def _cifrar(texto: str) -> str:
    try:   return _get_fernet().encrypt(texto.encode()).decode()
    except Exception: return texto

def _decifrar(cifrado: str) -> str:
    try:   return _get_fernet().decrypt(cifrado.encode()).decode()
    except Exception: return cifrado


# ============================================================================
# CONFIGURAÇÃO DE CONEXÃO
# ============================================================================

@dataclass
class ConfigConexao:
    nome:         str
    tipo:         str
    host:         str  = "localhost"
    porta:        int  = 3306
    banco:        str  = ""
    usuario:      str  = ""
    senha:        str  = ""
    # SSH tunnel
    usar_ssh:     bool = False
    ssh_host:     str  = ""
    ssh_porta:    int  = 22
    ssh_usuario:  str  = ""
    ssh_senha:    str  = ""
    ssh_key_path: str  = ""
    extras:       dict = field(default_factory=dict)


# ============================================================================
# PERSISTÊNCIA DE CONEXÕES EM DISCO
# ============================================================================

def salvar_conexoes_disco() -> None:
    """Salva configs com senha criptografada (Fernet)."""
    _CACHE_DIR.mkdir(exist_ok=True)
    dados = {}
    for nome, cfg in st.session_state.get("db_configs", {}).items():
        d = asdict(cfg)
        d["senha"]    = _cifrar(d["senha"])
        d["ssh_senha"] = _cifrar(d["ssh_senha"])
        dados[nome] = d
    _CONN_FILE.write_text(json.dumps(dados, indent=2, ensure_ascii=False))
    logger.info("Conexões salvas em disco (%d).", len(dados))


def carregar_conexoes_disco() -> None:
    """Restaura configs do disco para session_state (sem reconectar)."""
    if not _CONN_FILE.exists():
        return
    try:
        dados = json.loads(_CONN_FILE.read_text(encoding="utf-8"))
        for nome, d in dados.items():
            d["senha"]    = _decifrar(d["senha"])
            d["ssh_senha"] = _decifrar(d["ssh_senha"])
            cfg = ConfigConexao(**{k: v for k, v in d.items() if k in ConfigConexao.__dataclass_fields__})
            st.session_state.db_configs[nome] = cfg
        logger.info("Conexões carregadas do disco (%d).", len(dados))
    except Exception as e:
        logger.warning("Falha ao carregar conexões: %s", e)


# ============================================================================
# ESTADO DA SESSÃO
# ============================================================================

def inicializar_estado_db() -> None:
    defaults = {
        "db_configs":          {},
        "db_engines":          {},
        "db_tunnels":          {},   # {nome: SSHTunnelForwarder}
        "db_ativo":            None,
        "db_schema_map":       {},
        "db_chat_history":     [],
        "db_query_manual":     "",
        "db_resultado_manual": None,
        "db_query_history":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Carrega conexões salvas e schemas em cache na primeira inicialização
    if not st.session_state.db_configs:
        carregar_conexoes_disco()
        for nome in list(st.session_state.db_configs.keys()):
            carregar_cache_schema(nome)


# ============================================================================
# VALIDADOR SQL
# ============================================================================

_RE_BLOQ   = re.compile(r"^\s*(DROP|TRUNCATE|ALTER\s+TABLE|CREATE\s+TABLE|GRANT|REVOKE)\b", re.I|re.M)
_RE_DEL_NW = re.compile(r"\bDELETE\s+FROM\s+\S+\s*(?:;|$)", re.I)
_RE_UPD_NW = re.compile(r"\bUPDATE\s+\S+\s+SET\b(?:(?!\bWHERE\b).)*(?:;|$)", re.I|re.S)
_RE_RO     = re.compile(r"^\s*(SELECT|WITH|SHOW|EXPLAIN|DESCRIBE|PRAGMA)\b", re.I)

def _validar_sql(sql: str, modo: str = "manual") -> tuple[bool, list[str]]:
    erros: list[str] = []
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    if len(stmts) > 1 and modo == "llm":
        return False, ["❌ Multi-statement bloqueado no modo automático."]
    for stmt in stmts:
        if _RE_BLOQ.match(stmt):
            return False, [f"❌ Comando '{stmt.split()[0].upper()}' bloqueado por segurança."]
        if modo == "llm" and not _RE_RO.match(stmt):
            return False, ["❌ Apenas consultas de leitura (SELECT/WITH/SHOW) permitidas no modo automático."]
        if modo != "llm":
            if _RE_DEL_NW.search(stmt): return False, ["❌ DELETE sem WHERE bloqueado."]
            if _RE_UPD_NW.search(stmt): return False, ["❌ UPDATE sem WHERE bloqueado."]
    return True, erros


# ============================================================================
# CONEXÃO
# ============================================================================

def _build_url(cfg: ConfigConexao, override_host: str = "", override_porta: int = 0) -> str:
    import urllib.parse
    if cfg.tipo == "SQLite":
        return f"sqlite:///{cfg.banco.strip() or ':memory:'}"
    h = override_host or cfg.host
    p = override_porta or cfg.porta
    porta = f":{p}" if p else ""
    cred  = ""
    if cfg.usuario:
        cred = f"{urllib.parse.quote_plus(cfg.usuario)}:{urllib.parse.quote_plus(cfg.senha)}@"
    return f"{BANCOS_SUPORTADOS[cfg.tipo]['driver']}://{cred}{h}{porta}/{cfg.banco}"


def _iniciar_ssh_tunnel(cfg: ConfigConexao):
    """Abre SSH tunnel e retorna (tunnel, local_host, local_port)."""
    try:
        from sshtunnel import SSHTunnelForwarder
        kw = dict(
            ssh_username=cfg.ssh_usuario,
            remote_bind_address=(cfg.host, cfg.porta),
        )
        if cfg.ssh_key_path and os.path.exists(cfg.ssh_key_path):
            kw["ssh_pkey"] = cfg.ssh_key_path
        elif cfg.ssh_senha:
            kw["ssh_password"] = cfg.ssh_senha
        tunnel = SSHTunnelForwarder((cfg.ssh_host, cfg.ssh_porta), **kw)
        tunnel.start()
        logger.info("SSH tunnel aberto: %s:%d → 127.0.0.1:%d", cfg.ssh_host, cfg.ssh_porta, tunnel.local_bind_port)
        return tunnel, "127.0.0.1", tunnel.local_bind_port
    except Exception as e:
        raise RuntimeError(f"Falha no SSH tunnel: {e}")


def conectar(cfg: ConfigConexao) -> tuple[bool, str]:
    tunnel = local_host = local_port = None
    try:
        if cfg.usar_ssh and cfg.ssh_host:
            tunnel, local_host, local_port = _iniciar_ssh_tunnel(cfg)

        kwargs = {"pool_pre_ping": True}
        if cfg.tipo != "SQLite":
            kwargs["connect_args"] = {"connect_timeout": 10}

        url    = _build_url(cfg, local_host or "", local_port or 0)
        engine = create_engine(url, **kwargs)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        st.session_state.db_configs[cfg.nome] = cfg
        st.session_state.db_engines[cfg.nome] = engine
        if tunnel:
            st.session_state.db_tunnels[cfg.nome] = tunnel
        st.session_state.db_ativo = cfg.nome
        salvar_conexoes_disco()
        logger.info("Conectado: %s (%s)", cfg.nome, cfg.tipo)
        return True, f"Conectado a **{cfg.nome}** ({cfg.tipo})"

    except Exception as e:
        if tunnel:
            try: tunnel.stop()
            except Exception: pass
        return False, f"Erro: {_limpar_erro(str(e))}"


def desconectar(nome: str) -> None:
    eng = st.session_state.db_engines.pop(nome, None)
    if eng: eng.dispose()
    tunnel = st.session_state.db_tunnels.pop(nome, None)
    if tunnel:
        try: tunnel.stop()
        except Exception: pass
    st.session_state.db_configs.pop(nome, None)
    st.session_state.db_schema_map.pop(nome, None)
    salvar_conexoes_disco()
    if st.session_state.db_ativo == nome:
        restantes = list(st.session_state.db_engines.keys())
        st.session_state.db_ativo = restantes[0] if restantes else None


# ============================================================================
# MAPEAMENTO DE SCHEMA
# ============================================================================

def mapear_banco(nome: str) -> tuple[bool, str]:
    engine = st.session_state.db_engines.get(nome)
    if not engine:
        return False, "Engine não encontrado."
    cfg  = st.session_state.db_configs[nome]
    insp = sa_inspect(engine)
    schema_map = {"tipo": cfg.tipo, "banco": cfg.banco, "tabelas": {}, "views": {}, "procedures": []}

    try: tabelas = sorted(insp.get_table_names())
    except Exception as e: return False, f"Erro ao listar tabelas: {e}"

    prog  = st.progress(0, text="Mapeando tabelas…")
    total = max(len(tabelas), 1)

    for i, tab in enumerate(tabelas):
        prog.progress((i+1)/total, text=f"Mapeando {tab}…")
        info = {"colunas":[],"pks":[],"fks":[],"indices":[],"row_count":None,"sample_data":[]}
        try: info["colunas"] = [{"nome":c["name"],"tipo":str(c["type"]),"nullable":c.get("nullable",True),"default":str(c.get("default") or "")} for c in insp.get_columns(tab)]
        except Exception: pass
        try: info["pks"] = insp.get_pk_constraint(tab).get("constrained_columns",[])
        except Exception: pass
        try: info["fks"] = [{"colunas_locais":fk.get("constrained_columns",[]),"tabela_ref":fk.get("referred_table",""),"colunas_ref":fk.get("referred_columns",[])} for fk in insp.get_foreign_keys(tab)]
        except Exception: pass
        try: info["indices"] = [{"nome":idx.get("name",""),"colunas":idx.get("column_names",[]),"unique":idx.get("unique",False)} for idx in insp.get_indexes(tab)]
        except Exception: pass
        try:
            with engine.connect() as conn: info["row_count"] = conn.execute(text(f"SELECT COUNT(*) FROM {tab}")).scalar()
        except Exception: info["row_count"] = -1
        try:
            with engine.connect() as conn:
                res = conn.execute(text(f"SELECT * FROM {tab} LIMIT {_SAMPLE_ROWS}"))
                info["sample_data"] = [dict(zip(list(res.keys()),[str(v) for v in r])) for r in res.fetchall()]
        except Exception: pass
        schema_map["tabelas"][tab] = info

    # Views
    try:
        for view in sorted(insp.get_view_names()):
            v_info = {"colunas":[], "tipo":"view"}
            try: v_info["colunas"] = [{"nome":c["name"],"tipo":str(c["type"])} for c in insp.get_columns(view)]
            except Exception: pass
            schema_map["views"][view] = v_info
    except Exception: pass

    # Stored Procedures (MySQL / PostgreSQL)
    try:
        with engine.connect() as conn:
            if cfg.tipo in ("MySQL / MariaDB",):
                rows = conn.execute(text("SELECT ROUTINE_NAME FROM information_schema.ROUTINES WHERE ROUTINE_TYPE='PROCEDURE' AND ROUTINE_SCHEMA=DATABASE()")).fetchall()
                schema_map["procedures"] = [r[0] for r in rows]
            elif cfg.tipo == "PostgreSQL":
                rows = conn.execute(text("SELECT proname FROM pg_proc JOIN pg_namespace ON pg_namespace.oid=pg_proc.pronamespace WHERE nspname='public'")).fetchall()
                schema_map["procedures"] = [r[0] for r in rows]
    except Exception: pass

    prog.empty()
    st.session_state.db_schema_map[nome] = schema_map
    salvar_cache_schema(nome)
    return True, f"Mapeado: {len(schema_map['tabelas'])} tabela(s), {len(schema_map['views'])} view(s), {len(schema_map['procedures'])} procedure(s)."


# ============================================================================
# CACHE DE SCHEMA
# ============================================================================

def _cache_path(nome: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"schema_{re.sub(r'[^a-zA-Z0-9_-]','_',nome)}.json"

def salvar_cache_schema(nome: str) -> None:
    s = st.session_state.db_schema_map.get(nome)
    if s: _cache_path(nome).write_text(json.dumps(s,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def carregar_cache_schema(nome: str) -> bool:
    p = _cache_path(nome)
    if not p.exists(): return False
    try:
        st.session_state.db_schema_map[nome] = json.loads(p.read_text(encoding="utf-8"))
        return True
    except Exception: return False


# ============================================================================
# CONTEXTO TEXTUAL DO SCHEMA + DIAGRAMA ER
# ============================================================================

def gerar_contexto_schema(nome: str) -> str:
    sm = st.session_state.db_schema_map.get(nome)
    if not sm: return "(schema não mapeado)"
    cfg   = st.session_state.db_configs.get(nome)
    lines = [f"=== SCHEMA: {sm['tipo']} | {sm['banco'] or nome} ===",""]
    for tab, info in sm["tabelas"].items():
        c = info.get("row_count",-1)
        lines.append(f"TABELA: {tab}  [{f'{c:,}' if isinstance(c,int) and c>=0 else '?'} linhas]")
        for col in info.get("colunas",[]):
            flags = (["PK"] if col["nome"] in info.get("pks",[]) else []) + [f"FK→{fk['tabela_ref']}" for fk in info.get("fks",[]) if col["nome"] in fk.get("colunas_locais",[])]
            lines.append(f"  {col['nome']:<25} {col['tipo']:<18} {'NULL' if col['nullable'] else 'NOT NULL'}  {', '.join(flags)}")
        for fk in info.get("fks",[]): lines.append(f"  FK: {tab}.{', '.join(fk['colunas_locais'])} → {fk['tabela_ref']}.{', '.join(fk['colunas_ref'])}")
        if info.get("sample_data"):
            lines.append("  Amostra:")
            lines.append("    " + " | ".join(f"{k[:12]}" for k in info["sample_data"][0].keys()))
            for row in info["sample_data"]: lines.append("    " + " | ".join(f"{str(v)[:12]}" for v in row.values()))
        lines.append("")
    if sm.get("views"): lines.append(f"VIEWS: {', '.join(sm['views'].keys())}")
    if sm.get("procedures"): lines.append(f"STORED PROCEDURES: {', '.join(sm['procedures'])}")
    texto = "\n".join(lines)
    if len(texto) > _MAX_SCHEMA_CHARS: texto = texto[:_MAX_SCHEMA_CHARS] + "\n[... truncado ...]"
    return texto


def gerar_diagrama_er(nome: str) -> str:
    sm  = st.session_state.db_schema_map.get(nome, {})
    tabs = sm.get("tabelas", {})
    def esc(s): return str(s).replace('"',"'").replace("{","").replace("}","").replace("<","").replace(">","").replace("|","∣")
    dot = ['digraph ER {','  rankdir=LR;','  graph [fontsize=10 fontname=Helvetica];','  node [shape=record fontsize=9 style="filled,rounded" fillcolor="#FFF9C4" color="#888"];','  edge [fontsize=8 color=navy arrowhead=crow arrowtail=none dir=both];','']
    for tab, info in tabs.items():
        safe = re.sub(r"\W","_",tab)
        c    = info.get("row_count",-1); cs = f"{c:,}" if isinstance(c,int) and c>=0 else "?"
        cols = ""
        for col in info.get("colunas",[])[:14]:
            flags = (["PK"] if col["nome"] in info.get("pks",[]) else []) + (["FK"] if any(col["nome"] in fk.get("colunas_locais",[]) for fk in info.get("fks",[])) else [])
            tipo  = esc(str(col["tipo"]).split("(")[0][:8])
            null  = "" if col.get("nullable") else "✱"
            flag_s= f"[{','.join(flags)}]" if flags else ""
            cols += f"{esc(col['nome'])}{null}: {tipo} {flag_s}\\l"
        if len(info.get("colunas",[])) > 14: cols += f"... +{len(info['colunas'])-14}\\l"
        dot.append(f'  {safe} [label="{{{esc(tab)} ({cs})|{cols}}}"];')
    dot.append("")
    for tab, info in tabs.items():
        sf = re.sub(r"\W","_",tab)
        for fk in info.get("fks",[]):
            st2 = re.sub(r"\W","_",fk.get("tabela_ref",""))
            if st2 in [re.sub(r"\W","_",t) for t in tabs]:
                dot.append(f'  {sf} -> {st2} [label="{fk.get("colunas_locais",[""])[0]}"];')
    dot.append("}")
    return "\n".join(dot)


# ============================================================================
# HISTÓRICO + PAGINAÇÃO
# ============================================================================

def _registrar_hist(sql:str, rows, conn:str) -> None:
    h = st.session_state.setdefault("db_query_history",[])
    h.insert(0,{"sql":sql,"ts":datetime.datetime.now().strftime("%H:%M:%S"),"rows":rows if rows is not None else "-","conn":conn})
    st.session_state.db_query_history = h[:_MAX_HIST]

def _paginador(df:pd.DataFrame, key:str) -> pd.DataFrame:
    if len(df) <= _ROWS_PER_PAGE: return df
    tp = (len(df)+_ROWS_PER_PAGE-1)//_ROWS_PER_PAGE
    c1,c2 = st.columns([0.3,0.7])
    pg = c1.number_input(f"Página (1–{tp})",1,tp,1,key=f"pg_{key}")
    c2.caption(f"Exibindo {_ROWS_PER_PAGE} de **{len(df):,}** linhas")
    s = (pg-1)*_ROWS_PER_PAGE
    return df.iloc[s:s+_ROWS_PER_PAGE]


# ============================================================================
# EXECUÇÃO
# ============================================================================

def executar_query(sql:str, nome_conexao:str|None=None, modo:str="manual") -> tuple[bool,str,pd.DataFrame|None]:
    nome   = nome_conexao or st.session_state.get("db_ativo")
    engine = st.session_state.db_engines.get(nome) if nome else None
    if not engine: return False,"Nenhuma conexão ativa.",None
    sql = sql.strip()
    if not sql: return False,"Query vazia.",None
    ok, msgs = _validar_sql(sql, modo)
    if not ok: return False,"\n".join(msgs),None
    for m in [x for x in msgs if x.startswith("⚠️")]: st.warning(m)
    try:
        with engine.connect() as conn:
            r = conn.execute(text(sql))
            if r.returns_rows:
                df = pd.DataFrame(r.fetchall(), columns=list(r.keys()))
                _registrar_hist(sql,len(df),nome)
                return True,f"✅ {len(df):,} linha(s).",df
            conn.commit(); n=r.rowcount
            _registrar_hist(sql,n,nome)
            return True,f"✅ {n if n>=0 else '?'} linha(s) afetada(s).",None
    except SQLAlchemyError as e: return False,f"❌ {_limpar_erro(str(e))}",None
    except Exception as e:       return False,f"❌ {e}",None


# ============================================================================
# NL-to-SQL
# ============================================================================

def _extrair_sql(t:str)->str:
    t = re.sub(r"```(?:sql)?\s*","",t,flags=re.I).replace("```","").strip()
    return "\n".join(l for l in t.splitlines() if l.strip())

def _df_txt(df:pd.DataFrame)->str:
    if df is None or df.empty: return "(nenhum resultado)"
    t = df.head(_MAX_RESULT_LLM).to_string(index=False,max_colwidth=40)
    return t + (f"\n[...{len(df)-_MAX_RESULT_LLM} omitidas...]" if len(df)>_MAX_RESULT_LLM else "")

def responder_pergunta_banco(pergunta:str, nome:str) -> dict:
    llm = st.session_state.get("llm")
    if not llm: return {"erro":"LLM não inicializado."}
    schema = gerar_contexto_schema(nome)
    if "(schema não mapeado)" in schema: return {"erro":"Schema não mapeado. Clique em 🗺️ Mapear Banco."}
    cfg     = st.session_state.db_configs.get(nome)
    dialeto = BANCOS_SUPORTADOS[cfg.tipo]["dialect_hint"] if cfg else "SQL"
    try:
        r = llm.invoke([SystemMessage(content=f"Especialista em {dialeto}. Gere APENAS SQL puro (sem markdown) para a pergunta. Use somente tabelas do schema. Limite 200 linhas. NUNCA gere DROP/DELETE/UPDATE/INSERT.\n{schema}"), HumanMessage(content=pergunta)])
        sql = _extrair_sql(r.content)
    except Exception as e: return {"erro":f"Falha ao gerar SQL: {e}"}
    if not sql: return {"erro":"LLM não retornou SQL válido."}
    ok,msg,df = executar_query(sql,nome,modo="llm")
    if not ok: return {"sql":sql,"ok":False,"msg":msg,"df":None,"resposta":None,"erro":msg}
    resultado = _df_txt(df) if df is not None else "(sem retorno)"
    try:
        rf = llm.invoke([SystemMessage(content="Analista de dados. Responda em português com base nos resultados. Não explique o SQL."), HumanMessage(content=f"Pergunta: {pergunta}\nSQL:\n{sql}\nResultado:\n{resultado}")])
        resposta = rf.content
    except Exception as e: resposta = f"(Erro ao interpretar: {e})"
    return {"sql":sql,"ok":True,"msg":msg,"df":df,"resposta":resposta,"erro":None}


# ============================================================================
# EXPORT + HELPERS
# ============================================================================

def exportar_para_rag(df:pd.DataFrame, nome:str="query") -> tuple[bool,str]:
    from data_processing import adicionar_ao_indice
    try:
        os.makedirs("dados_docs",exist_ok=True)
        c = os.path.join("dados_docs",f"db_{nome}.csv")
        df.to_csv(c,index=False,encoding="utf-8")
        return adicionar_ao_indice([c])
    except Exception as e: return False,f"Erro: {e}"

def _limpar_erro(msg:str)->str:
    return " ".join(l for l in msg.splitlines()[:3] if not l.strip().startswith("(Background"))

def _botoes_dl(df:pd.DataFrame, prefix:str) -> None:
    c1,c2,c3 = st.columns(3)
    c1.download_button("⬇️ CSV",df.to_csv(index=False).encode(),"resultado.csv","text/csv",use_container_width=True,key=f"{prefix}_csv")
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w: df.to_excel(w,index=False)
    c2.download_button("⬇️ Excel",buf.getvalue(),"resultado.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True,key=f"{prefix}_xl")
    if c3.button("📚 → RAG",use_container_width=True,key=f"{prefix}_rag"):
        ok,m = exportar_para_rag(df,prefix)
        st.success(m) if ok else st.error(m)


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def painel_conexao_db() -> None:
    inicializar_estado_db()
    conns = st.session_state.db_configs
    if conns:
        st.subheader("🔌 Conexões")
        for nome,cfg in list(conns.items()):
            icone  = BANCOS_SUPORTADOS[cfg.tipo]["icone"]
            ativo  = st.session_state.db_ativo==nome
            mapeado = nome in st.session_state.db_schema_map
            ssh_tag = " 🔒" if cfg.usar_ssh else ""
            c1,c2,c3 = st.columns([0.55,0.25,0.20])
            c1.markdown(f"**{icone} {nome}**{'🗺️' if mapeado else ''}{ssh_tag}{' ●' if ativo else ''}")
            if not ativo:
                if c2.button("Usar",key=f"usar_{nome}",use_container_width=True):
                    st.session_state.db_ativo=nome; carregar_cache_schema(nome); st.rerun()
            else: c2.success("Ativa",icon="✅")
            if c3.button("✕",key=f"desc_{nome}",use_container_width=True): desconectar(nome); st.rerun()
        st.divider()

    with st.expander("➕ Nova Conexão", expanded=not bool(conns)):
        tipo   = st.selectbox("Banco",list(BANCOS_SUPORTADOS.keys()),format_func=lambda t:f"{BANCOS_SUPORTADOS[t]['icone']}  {t}",key="db_form_tipo")
        info   = BANCOS_SUPORTADOS[tipo]
        nome_c = st.text_input("Nome",placeholder="ex: producao",key="db_form_nome")
        if not info["requer_host"]:
            banco=st.text_input("Arquivo .db",key="db_sqlite"); host,porta,usuario,senha="",None,"",""
        else:
            ch,cp=st.columns([0.7,0.3]); host=ch.text_input("Host",value="localhost",key="db_form_host"); porta=cp.number_input("Porta",value=info["porta_padrao"],min_value=1,max_value=65535,key="db_form_porta")
            banco=st.text_input("Banco",key="db_form_banco"); usuario=st.text_input("Usuário",key="db_form_usuario"); senha=st.text_input("Senha",type="password",key="db_form_senha")
        usar_ssh = st.checkbox("🔒 Usar SSH Tunnel",key="db_ssh_toggle")
        ssh_host=ssh_usuario=ssh_senha=ssh_key=""; ssh_porta=22
        if usar_ssh:
            sh1,sh2=st.columns([0.7,0.3]); ssh_host=sh1.text_input("SSH Host",key="ssh_h"); ssh_porta=sh2.number_input("SSH Porta",value=22,min_value=1,max_value=65535,key="ssh_p")
            ssh_usuario=st.text_input("SSH Usuário",key="ssh_u"); ssh_senha=st.text_input("SSH Senha",type="password",key="ssh_pw"); ssh_key=st.text_input("Caminho chave privada (opcional)",key="ssh_k",placeholder="/home/user/.ssh/id_rsa")
        if st.button("🔗 Conectar",type="primary",use_container_width=True):
            if not nome_c.strip(): st.error("Defina um nome.")
            elif nome_c in st.session_state.db_configs: st.error(f"Já existe '{nome_c}'.")
            else:
                cfg = ConfigConexao(nome=nome_c.strip(),tipo=tipo,host=host,porta=int(porta or 0),banco=banco,usuario=usuario,senha=senha,usar_ssh=usar_ssh,ssh_host=ssh_host,ssh_porta=int(ssh_porta),ssh_usuario=ssh_usuario,ssh_senha=ssh_senha,ssh_key_path=ssh_key)
                with st.spinner("Conectando…"):
                    ok,msg=conectar(cfg)
                if ok: carregar_cache_schema(nome_c.strip()); st.success(msg); st.rerun()
                else: st.error(msg)


def pagina_banco_dados() -> None:
    inicializar_estado_db()
    st.title("🗄️ Banco de Dados")
    st.caption("Mapeie, consulte em linguagem natural, execute SQL e visualize relacionamentos.")
    conns = st.session_state.db_configs
    if not conns:
        st.info("Nenhuma conexão. Configure na aba **🗄️ Banco** da barra lateral.")
        return
    nomes = list(conns.keys())
    idx   = nomes.index(st.session_state.db_ativo) if st.session_state.db_ativo in nomes else 0
    sel   = st.selectbox("Conexão",nomes,index=idx,format_func=lambda n:f"{BANCOS_SUPORTADOS[conns[n].tipo]['icone']}  {n}  ({conns[n].tipo})")
    if sel!=st.session_state.db_ativo:
        st.session_state.db_ativo=sel; st.session_state.db_resultado_manual=None; carregar_cache_schema(sel); st.rerun()
    cfg_a   = conns[st.session_state.db_ativo]
    mapeado = st.session_state.db_ativo in st.session_state.db_schema_map
    ci,cm,cs = st.columns([0.55,0.25,0.20])
    ci.caption(f"**Host:** {cfg_a.host or 'local'}  |  **Banco:** {cfg_a.banco or '—'}  |  **User:** {cfg_a.usuario or '—'}" + (" 🔒SSH" if cfg_a.usar_ssh else ""))
    if cm.button("🗺️ Mapear",use_container_width=True,type="primary" if not mapeado else "secondary"):
        with st.spinner("Mapeando schema…"):
            ok,msg=mapear_banco(st.session_state.db_ativo)
        st.success(msg) if ok else st.error(msg)
        if ok: st.rerun()
    if mapeado:
        n=len(st.session_state.db_schema_map[st.session_state.db_ativo]["tabelas"])
        cs.success(f"✅ {n} tabelas")
    else: cs.warning("⚠️ Não mapeado")
    st.divider()

    tab_chat,tab_schema,tab_er,tab_sql,tab_hist = st.tabs(["💬 Chat NL","🗺️ Schema","🔗 Diagrama ER","✏️ SQL Manual","📋 Histórico"])

    with tab_chat:
        if not mapeado: st.info("Mapeie o banco primeiro.")
        elif not st.session_state.get("llm"): st.warning("Configure o LLM na aba 🤖 Modelo.")
        else:
            for entry in st.session_state.db_chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
                    if entry.get("sql"):
                        with st.expander("🔍 SQL gerado",expanded=False): st.code(entry["sql"],language="sql")
                    if entry.get("df") is not None and not entry["df"].empty:
                        with st.expander(f"📊 {len(entry['df']):,} linhas",expanded=False):
                            st.dataframe(_paginador(entry["df"],f"h{id(entry)}"),use_container_width=True,hide_index=True)
            if perg:=st.chat_input("Ex: Quais clientes compraram mais de R$1000 este mês?"):
                with st.chat_message("user"): st.markdown(perg)
                st.session_state.db_chat_history.append({"role":"user","content":perg,"sql":None,"df":None})
                with st.chat_message("assistant"):
                    with st.spinner("Gerando SQL…"): res=responder_pergunta_banco(perg,st.session_state.db_ativo)
                    if res.get("erro") and not res.get("sql"): st.error(res["erro"]); st.session_state.db_chat_history.append({"role":"assistant","content":res["erro"],"sql":None,"df":None})
                    else:
                        md=res.get("resposta") or res.get("erro") or "Sem resposta."
                        st.markdown(md)
                        if res.get("sql"):
                            with st.expander("🔍 SQL",expanded=False): st.code(res["sql"],language="sql")
                        df=res.get("df")
                        if df is not None and not df.empty:
                            with st.expander(f"📊 {len(df):,} linha(s)",expanded=True):
                                st.dataframe(_paginador(df,f"nl{len(st.session_state.db_chat_history)}"),use_container_width=True,hide_index=True)
                                _botoes_dl(df,f"nl{len(st.session_state.db_chat_history)}")
                        st.session_state.db_chat_history.append({"role":"assistant","content":md,"sql":res.get("sql"),"df":df})
            if st.session_state.db_chat_history:
                if st.button("🗑️ Limpar",use_container_width=True): st.session_state.db_chat_history=[]; st.rerun()

    with tab_schema:
        if not mapeado: st.info("Mapeie o banco.")
        else:
            sm = st.session_state.db_schema_map[st.session_state.db_ativo]
            busca = st.text_input("🔍 Filtrar",placeholder="Nome ou parte…")
            for secao,itens,tipo_label in [("📋 Tabelas",sm.get("tabelas",{}),"tabela"),("👁️ Views",sm.get("views",{}),"view")]:
                filtrados = {t:v for t,v in itens.items() if not busca or busca.lower() in t.lower()}
                if not filtrados: continue
                st.subheader(f"{secao} ({len(filtrados)})")
                for tab,info in filtrados.items():
                    c = info.get("row_count",-1); cs2 = f"{c:,}" if isinstance(c,int) and c>=0 else "?"
                    with st.expander(f"**{tab}**" + (f" — {cs2} linhas" if tipo_label=="tabela" else ""), expanded=False):
                        if info.get("colunas"):
                            cols_df=[{"Coluna":c2["nome"],"Tipo":c2["tipo"],"Nulo":"✓" if c2.get("nullable") else "✗","Flags":" ".join(["🔑PK"] if c2["nome"] in info.get("pks",[]) else [])+" ".join(f"🔗FK→{fk['tabela_ref']}" for fk in info.get("fks",[]) if c2["nome"] in fk.get("colunas_locais",[]))} for c2 in info["colunas"]]
                            st.dataframe(pd.DataFrame(cols_df),use_container_width=True,hide_index=True)
                        for fk in info.get("fks",[]): st.markdown(f"🔗 `{tab}.{', '.join(fk['colunas_locais'])}` → `{fk['tabela_ref']}.{', '.join(fk['colunas_ref'])}`")
                        if info.get("sample_data"):
                            with st.expander("Amostra",expanded=False): st.dataframe(pd.DataFrame(info["sample_data"]),use_container_width=True,hide_index=True)
                        if tipo_label=="tabela" and st.button(f"▶ SELECT * FROM {tab} LIMIT 100",key=f"sel_{tab}",use_container_width=True):
                            st.session_state.db_query_manual=f"SELECT *\nFROM   {tab}\nLIMIT  100;"; st.rerun()
            if sm.get("procedures"):
                st.subheader(f"⚙️ Procedures ({len(sm['procedures'])})")
                for p in sm["procedures"]: st.code(p)

    with tab_er:
        if not mapeado: st.info("Mapeie o banco para gerar o diagrama.")
        else:
            tabs2 = st.session_state.db_schema_map[st.session_state.db_ativo]["tabelas"]
            if len(tabs2)>60: st.warning(f"⚠️ {len(tabs2)} tabelas — diagrama pode ficar grande.")
            dot = gerar_diagrama_er(st.session_state.db_ativo)
            st.graphviz_chart(dot,use_container_width=True)
            st.download_button("⬇️ DOT",dot.encode(),"er.dot","text/plain",use_container_width=True)

    with tab_sql:
        query = st.text_area("SQL",value=st.session_state.db_query_manual,height=200,placeholder="SELECT * FROM tabela LIMIT 100;",label_visibility="collapsed",key="sql_ed")
        st.session_state.db_query_manual = query
        c1,c2,c3 = st.columns([0.2,0.2,0.6])
        run   = c1.button("▶ Executar",type="primary",use_container_width=True)
        explain = c2.button("📋 Explain",use_container_width=True,help="Mostra plano de execução")
        if c3.button("🗑 Limpar",use_container_width=True): st.session_state.db_query_manual=""; st.session_state.db_resultado_manual=None; st.rerun()
        if explain and query.strip():
            cfg_a2 = st.session_state.db_configs.get(st.session_state.db_ativo)
            explain_sql = f"EXPLAIN ANALYZE {query}" if cfg_a2 and cfg_a2.tipo=="PostgreSQL" else f"EXPLAIN {query}"
            ok2,msg2,df2 = executar_query(explain_sql,modo="manual")
            if ok2 and df2 is not None:
                st.subheader("📋 Plano de Execução")
                st.dataframe(df2,use_container_width=True,hide_index=True)
            else: st.error(msg2)
        if run:
            if not query.strip(): st.warning("Digite um SQL.")
            else:
                with st.spinner("Executando…"): ok2,msg2,df2=executar_query(query)
                st.session_state.db_resultado_manual=df2 if ok2 else None
                st.success(msg2) if ok2 else st.error(msg2)
        df_m = st.session_state.get("db_resultado_manual")
        if df_m is not None:
            st.subheader(f"📊 {len(df_m):,} × {len(df_m.columns)} col(s)")
            st.dataframe(_paginador(df_m,"man"),use_container_width=True,hide_index=True)
            _botoes_dl(df_m,"man")

    with tab_hist:
        hist=st.session_state.get("db_query_history",[])
        if not hist: st.info("Nenhuma query executada.")
        else:
            for i,e in enumerate(hist):
                with st.expander(f"🕐 {e['ts']}  |  {e['conn']}  |  {e['rows']} linhas",expanded=False):
                    st.code(e["sql"],language="sql")
                    if st.button("▶ Re-executar",key=f"re_{i}",use_container_width=True):
                        st.session_state.db_query_manual=e["sql"]; st.rerun()
            if st.button("🗑️ Limpar",use_container_width=True): st.session_state.db_query_history=[]; st.rerun()
