"""
app.py — PlenoDoc: ponto de entrada principal.

Funcionalidades desta versão:
  - Streaming, fontes com score, feedback, export, sugestões pós-resposta
  - Modo Comparar LLMs (dois modelos em paralelo)
  - Cache de respostas LLM, rate-limiting, memória persistente
  - Dashboard de métricas na sidebar
  - Painel de auditoria para admins
  - Resumos automáticos de documentos indexados
  - Configuração centralizada via config.py + .env
"""

import os, logging, datetime, time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud: propaga st.secrets para variáveis de ambiente
# para que pydantic-settings possa lê-las normalmente
def _sync_secrets_to_env() -> None:
    try:
        import streamlit as _st
        for key in (OPENAI_API_KEY, GROQ_API_KEY, SESSION_TIMEOUT_MINUTES,
                    MAX_REQUESTS_PER_MINUTE, ENABLE_RERANKING, ENABLE_HYBRID_SEARCH,
                    ENABLE_LLM_CACHE, LOG_LEVEL):
            val = _st.secrets.get(key)
            if val and not os.environ.get(key):
                os.environ[key] = str(val)
    except Exception:
        pass

_sync_secrets_to_env()

# ── Logging ──────────────────────────────────────────────────────────────────
def _setup_logging():
    os.makedirs("logs", exist_ok=True)
    fmt  = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    if root.handlers: return
    root.setLevel(logging.INFO)
    for h in [logging.FileHandler("logs/plenodoc.log","a","utf-8"), logging.StreamHandler()]:
        h.setFormatter(fmt); root.addHandler(h)

_setup_logging()
logger = logging.getLogger("plenodoc.app")

st.set_page_config(page_title="PlenoDoc", page_icon="📑", layout="wide", initial_sidebar_state="expanded")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import get_settings
from auth import pagina_login, verificar_timeout, tem_permissao, usuario_atual, painel_usuarios, verificar_rate_limit
from audit import registrar as audit, painel_audit_ui
from cache_llm import get_cached, set_cached, stats as cache_stats, limpar_expirado
from memory import salvar_mensagem, carregar_historico, limpar_historico as limpar_mem
from data_processing import (
    adicionar_ao_indice, reconstruir_indice, inicializar_retriever,
    adicionar_url_ao_indice, obter_resumos, gerar_resumo_documento,
)
from database import painel_conexao_db, pagina_banco_dados, inicializar_estado_db
from startup import run as _startup_run

MODELOS_DISPONIVEIS = {
    "Groq (Fast)":     {"versao_api":["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768"], "chat":ChatGroq},
    "OpenAI (Premium)":{"versao_api":["gpt-4o-mini","gpt-4o"],                                               "chat":ChatOpenAI},
}

# ── Estado da sessão ─────────────────────────────────────────────────────────
_DEFAULTS = {
    "logged_in":False,"username":"","user_role":"viewer","display_name":"",
    "last_activity":0.0,"rl_requests":[],
    "chat_history":[],"chat_feedback":{},"chat_sources":{},"chat_suggestions":{},
    "chain":None,"retriever":None,"llm":None,"llm_b":None,
    "modo_tela":"chat","comparar_llms":False,
    "llm_temperatura":0.1,"llm_max_tokens":1000,
    "llm_b_provedor":"Groq (Fast)","llm_b_modelo":"llama-3.3-70b-versatile","llm_b_key":"",
    "metricas":{"total":0,"tokens_est":0,"latencias":[],"fb_pos":0,"fb_neg":0,"cache_hits":0},
    "historico_carregado":False,
}
for k,v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

inicializar_estado_db()

# ── Utilitários ───────────────────────────────────────────────────────────────
def _criar_llm(provedor, modelo, api_key, temp, max_tok):
    return MODELOS_DISPONIVEIS[provedor]["chat"](model=modelo, api_key=api_key, temperature=temp, max_tokens=max_tok)

def _atualizar_metricas(latencia_s: float, tokens_est: int) -> None:
    m = st.session_state.metricas
    m["total"]      += 1
    m["tokens_est"] += tokens_est
    lats = m["latencias"]
    lats.append(latencia_s)
    if len(lats) > 100: m["latencias"] = lats[-100:]

def _exportar_chat() -> bytes:
    agora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    linhas = [f"# PlenoDoc — Chat\n_Exportado em {agora}_\n\n---\n"]
    for i,msg in enumerate(st.session_state.chat_history):
        role = "👤 **Usuário**" if isinstance(msg,HumanMessage) else "🤖 **PlenoDoc**"
        linhas.append(f"{role}:\n{msg.content}\n")
        fb = st.session_state.chat_feedback.get(i)
        if fb: linhas.append(f"_Avaliação: {'👍' if fb=='up' else '👎'}_\n")
        srcs = st.session_state.chat_sources.get(i,[])
        if srcs: linhas.append(f"_Fontes: {', '.join(srcs)}_\n")
        linhas.append("---\n")
    return "\n".join(linhas).encode("utf-8")


# ── Sugestões de perguntas ───────────────────────────────────────────────────

def _gerar_sugestoes(pergunta: str, resposta: str, idx: int) -> None:
    cfg = get_settings()
    if not cfg.enable_suggestions: return
    llm = st.session_state.get("llm")
    if not llm: return
    try:
        r = llm.invoke([
            SystemMessage(content="Gere exatamente 3 perguntas de follow-up curtas e relevantes em português. Retorne apenas as perguntas, uma por linha, sem numeração."),
            HumanMessage(content=f"Pergunta: {pergunta}\nResposta: {resposta[:400]}"),
        ])
        sugs = [l.strip() for l in r.content.splitlines() if len(l.strip()) > 10][:3]
        st.session_state.chat_suggestions[idx] = sugs
    except Exception: pass


# ============================================================================
# PÁGINA DE CHAT
# ============================================================================

def pagina_chat() -> None:
    st.title("PlenoDoc 📑")
    st.caption("Módulo de Consulta Automatizada de Suporte Técnico.")

    # Barra de ações
    if st.session_state.chat_history:
        c1,c2,c3,_ = st.columns([0.15,0.15,0.15,0.55])
        if c1.button("🗑️ Limpar",use_container_width=True):
            st.session_state.chat_history=[]; st.session_state.chat_feedback={}
            st.session_state.chat_sources={}; st.session_state.chat_suggestions={}
            if st.session_state.username: limpar_mem(st.session_state.username)
            st.rerun()
        c2.download_button("⬇️ Export",_exportar_chat(),f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md","text/markdown",use_container_width=True)
        comp_label = "🔀 Comparar ON" if st.session_state.comparar_llms else "🔀 Comparar"
        if c3.button(comp_label, use_container_width=True, type="primary" if st.session_state.comparar_llms else "secondary"):
            st.session_state.comparar_llms = not st.session_state.comparar_llms; st.rerun()

    st.divider()

    # Exibe histórico
    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if isinstance(msg,HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant":
                srcs = st.session_state.chat_sources.get(i,[])
                if srcs:
                    with st.expander(f"📎 Fontes ({len(srcs)})",expanded=False):
                        for s in sorted(srcs): st.markdown(f"• `{s}`")
                fb = st.session_state.chat_feedback.get(i)
                fc1,fc2,_ = st.columns([0.045,0.045,0.91])
                if fc1.button("👍",key=f"up_{i}",type="primary" if fb=="up" else "secondary"):
                    st.session_state.chat_feedback[i]="up"; st.session_state.metricas["fb_pos"]+=1; st.rerun()
                if fc2.button("👎",key=f"dw_{i}",type="primary" if fb=="down" else "secondary"):
                    st.session_state.chat_feedback[i]="down"; st.session_state.metricas["fb_neg"]+=1; st.rerun()
                sugs = st.session_state.chat_suggestions.get(i,[])
                if sugs:
                    st.caption("💡 Perguntas relacionadas:")
                    scols = st.columns(len(sugs))
                    for j,(col,sug) in enumerate(zip(scols,sugs)):
                        if col.button(sug,key=f"sug_{i}_{j}",use_container_width=True):
                            st.session_state["_sug_pending"] = sug; st.rerun()

    if not st.session_state.chain:
        st.info("ℹ️ LLM offline. Configure a API Key na barra lateral.")
        st.chat_input("Aguardando inicialização…",disabled=True)
        return

    # Pergunta pendente de sugestão (clique no botão de sugestão)
    prompt_override = st.session_state.pop("_sug_pending", None)

    if prompt := (prompt_override or st.chat_input("Faça sua pergunta…")):
        ok_rl, msg_rl = verificar_rate_limit()
        if not ok_rl:
            st.warning(msg_rl); return

        with st.chat_message("user"): st.markdown(prompt)

        # ── Modo normal ──────────────────────────────────────────────────
        if not st.session_state.comparar_llms:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_answer = ""; context_docs = []
                t0 = time.time()

                # Verifica cache
                model_name = getattr(st.session_state.llm, "model_name", "") or getattr(st.session_state.llm, "model", "")
                cached = get_cached(prompt, model_name)
                if cached:
                    full_answer = cached; placeholder.markdown(full_answer)
                    st.session_state.metricas["cache_hits"] += 1
                else:
                    try:
                        for chunk in st.session_state.chain.stream({"input":prompt,"chat_history":st.session_state.chat_history}):
                            if "context" in chunk: context_docs = chunk["context"]
                            if "answer"  in chunk:
                                full_answer += chunk["answer"]
                                placeholder.markdown(full_answer+"▌")
                        placeholder.markdown(full_answer)
                        set_cached(prompt, model_name, full_answer)
                    except Exception as e:
                        placeholder.error(f"❌ {e}"); logger.error("Chat error: %s",e); return

                lat = time.time()-t0
                _atualizar_metricas(lat, int(len(full_answer.split())*1.3))

                user_idx = len(st.session_state.chat_history)
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                ai_idx = len(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=full_answer))

                if st.session_state.username:
                    salvar_mensagem(st.session_state.username,"human",prompt)
                    salvar_mensagem(st.session_state.username,"ai",full_answer)

                # Fontes
                fontes = {os.path.basename(str(d.metadata.get("source",""))) for d in context_docs if d.metadata.get("source","")}
                if fontes:
                    st.session_state.chat_sources[ai_idx] = sorted(fontes)
                    with st.expander(f"📎 Fontes ({len(fontes)})",expanded=False):
                        for s in sorted(fontes): st.markdown(f"• `{s}`")

                audit("chat_pergunta",f"prompt={prompt[:80]}")
                _gerar_sugestoes(prompt, full_answer, ai_idx)

        # ── Modo comparação ──────────────────────────────────────────────
        else:
            _comparar_llms(prompt)


def _comparar_llms(prompt: str) -> None:
    """Executa a mesma pergunta em dois LLMs e exibe lado a lado."""
    llm_a = st.session_state.get("llm")
    llm_b = st.session_state.get("llm_b")
    if not llm_a or not llm_b:
        st.warning("Configure os dois modelos na sidebar para usar a comparação.")
        return

    col_a, col_b = st.columns(2)
    ma = getattr(llm_a,"model_name","") or getattr(llm_a,"model","Modelo A")
    mb = getattr(llm_b,"model_name","") or getattr(llm_b,"model","Modelo B")

    hist = st.session_state.chat_history
    for col, llm, label in [(col_a, llm_a, ma), (col_b, llm_b, mb)]:
        with col:
            st.caption(f"**{label}**")
            ph = st.empty(); resp_txt = ""
            try:
                for chunk in st.session_state.chain.stream({"input":prompt,"chat_history":hist}):
                    if "answer" in chunk:
                        resp_txt += chunk["answer"]
                        ph.markdown(resp_txt+"▌")
                ph.markdown(resp_txt)
            except Exception as e:
                ph.error(f"❌ {e}")

    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=f"**{ma}:** {resp_txt[:300]}…"))


# ============================================================================
# PAINEL DE DOCUMENTOS
# ============================================================================

def painel_documentos() -> None:
    st.subheader("Gestão de Base (RAG)")
    cfg = get_settings()

    arquivos = st.file_uploader("Arquivos",accept_multiple_files=True,type=["pdf","csv","txt","docx"],label_visibility="collapsed")
    if st.button("⬆️ Processar",use_container_width=True,disabled=not bool(arquivos)):
        os.makedirs(cfg.docs_path,exist_ok=True)
        caminhos=[]
        for arq in arquivos:
            dest=os.path.join(cfg.docs_path,arq.name)
            with open(dest,"wb") as f: f.write(arq.getbuffer())
            caminhos.append(dest)
        with st.spinner("Vetorizando…"):
            ok,msg=adicionar_ao_indice(caminhos)
        if ok: st.session_state.chain=None; st.success(msg); audit("doc_adicionado",f"arquivos={[os.path.basename(c) for c in caminhos]}"); st.rerun()
        else: st.error(msg)

    st.subheader("🌐 Indexar URL")
    url_input = st.text_input("URL",placeholder="https://docs.exemplo.com/pagina",label_visibility="collapsed")
    if st.button("⬆️ Carregar URL",use_container_width=True,disabled=not bool(url_input and url_input.startswith("http"))):
        with st.spinner("Carregando URL…"):
            ok,msg=adicionar_url_ao_indice(url_input.strip())
        if ok: st.session_state.chain=None; st.success(msg); audit("url_indexada",f"url={url_input}")
        else: st.error(msg)

    st.divider()
    st.subheader("📋 Repositório")
    docs=sorted(f for f in os.listdir(cfg.docs_path) if not f.startswith(".")) if os.path.exists(cfg.docs_path) else []
    resumos = obter_resumos()
    if not docs: st.info("Repositório vazio.")
    for nome in docs:
        c1,c2=st.columns([0.85,0.15])
        c1.text(nome)
        resumo = resumos.get(nome,"")
        if resumo: c1.caption(resumo)
        if c2.button("🗑️",key=f"del_{nome}",use_container_width=True):
            os.remove(os.path.join(cfg.docs_path,nome))
            restantes=[os.path.join(cfg.docs_path,f) for f in os.listdir(cfg.docs_path) if not f.startswith(".")]
            with st.spinner("Reconstruindo…"):
                ok,msg=reconstruir_indice(restantes)
            if ok: st.session_state.chain=None; audit("doc_removido",f"arquivo={nome}"); st.toast("✅ Removido."); st.rerun()
            else: st.error(msg)


# ============================================================================
# INICIALIZAÇÃO DO LLM
# ============================================================================

def inicializar_modelo(provedor,modelo,api_key,temp,max_tok) -> None:
    retriever = inicializar_retriever()
    if not retriever: st.error("⚠️ Base vazia. Processe documentos primeiro."); return
    try:
        llm = _criar_llm(provedor,modelo,api_key,temp,max_tok)
        st.session_state.llm = llm
        prompt_ctx = ChatPromptTemplate.from_messages([
            ("system","Reformule a pergunta do usuário de forma independente do histórico para otimizar a busca. Não responda — apenas reformule."),
            MessagesPlaceholder("chat_history"),("human","{input}"),
        ])
        prompt_resp = ChatPromptTemplate.from_messages([
            ("system","Você é o PlenoDoc, assistente especialista de suporte técnico.\nResponda APENAS com base no contexto documental abaixo.\nSeja objetivo. Se não encontrar, diga: 'Informação não encontrada na base.'\nNUNCA invente dados técnicos.\nCite o documento de origem quando possível.\n\n[CONTEXTO]\n{context}\n[/CONTEXTO]"),
            MessagesPlaceholder("chat_history"),("human","{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm,retriever,prompt_ctx)
        doc_chain       = create_stuff_documents_chain(llm,prompt_resp)
        st.session_state.chain = create_retrieval_chain(retriever_chain,doc_chain)
        logger.info("Modelo RAG inicializado: %s/%s",provedor,modelo)
        audit("modelo_inicializado",f"provedor={provedor} modelo={modelo}")
        st.success(f"✅ **{modelo}** com RAG pronto.")
    except Exception as e:
        logger.error("Falha init modelo: %s",e); st.error(f"❌ {e}")
        st.session_state.chain=None; st.session_state.llm=None

def inicializar_llm_standalone(provedor,modelo,api_key,temp,max_tok) -> None:
    try:
        st.session_state.llm = _criar_llm(provedor,modelo,api_key,temp,max_tok)
        audit("llm_standalone",f"provedor={provedor} modelo={modelo}")
        st.success(f"✅ **{modelo}** pronto para banco de dados.")
    except Exception as e:
        st.error(f"❌ {e}"); st.session_state.llm=None


# ============================================================================
# SIDEBAR
# ============================================================================

def _painel_metricas() -> None:
    m = st.session_state.metricas
    with st.expander("📊 Métricas da Sessão",expanded=False):
        c1,c2=st.columns(2)
        c1.metric("Perguntas",m["total"])
        c2.metric("Tokens est.",f"{m['tokens_est']:,}")
        lats = m.get("latencias",[])
        if lats:
            avg = sum(lats)/len(lats)
            c1.metric("Latência média",f"{avg:.1f}s")
        cs = cache_stats()
        c2.metric("Cache hits",f"{m['cache_hits']} / {cs['validos']} válidos")
        if m["fb_pos"]+m["fb_neg"] > 0:
            total_fb = m["fb_pos"]+m["fb_neg"]
            st.progress(m["fb_pos"]/total_fb, text=f"👍 {m['fb_pos']}  👎 {m['fb_neg']}")
        if st.button("🗑️ Limpar cache LLM",use_container_width=True):
            n=limpar_expirado(); st.toast(f"✅ {n} entrada(s) expirada(s) removidas.")


def _painel_modelo() -> None:
    provedor = st.selectbox("Provedor",MODELOS_DISPONIVEIS.keys(),label_visibility="collapsed")
    modelo   = st.selectbox("Modelo",  MODELOS_DISPONIVEIS[provedor]["versao_api"],label_visibility="collapsed")
    api_key  = st.text_input("API Key",type="password",label_visibility="collapsed",placeholder=f"Chave para {provedor}…")
    st.subheader("Parâmetros")
    temp    = st.slider("Temperatura",0.0,1.0,st.session_state.llm_temperatura,0.05)
    max_tok = st.slider("Max Tokens", 256,4096,st.session_state.llm_max_tokens, 128)
    st.session_state.llm_temperatura = temp
    st.session_state.llm_max_tokens  = max_tok
    c1,c2 = st.columns(2)
    if c1.button("🚀 Chat RAG",type="primary",use_container_width=True):
        if not api_key.strip(): st.error("Informe a API Key.")
        else: inicializar_modelo(provedor,modelo,api_key.strip(),temp,max_tok)
    if c2.button("🗄️ Só Banco",use_container_width=True):
        if not api_key.strip(): st.error("Informe a API Key.")
        else: inicializar_llm_standalone(provedor,modelo,api_key.strip(),temp,max_tok)
    if st.session_state.get("llm"):   st.success("LLM ativo ✅",icon="🤖")
    else: st.info("LLM não inicializado")
    if st.session_state.get("chain"): st.success("RAG ativo ✅",icon="📚")

    # Modelo B para comparação
    if st.session_state.comparar_llms:
        st.divider(); st.subheader("🔀 Modelo B (comparação)")
        prov_b  = st.selectbox("Provedor B",MODELOS_DISPONIVEIS.keys(),key="llm_b_prov_sel")
        mod_b   = st.selectbox("Modelo B",  MODELOS_DISPONIVEIS[prov_b]["versao_api"],key="llm_b_mod_sel")
        key_b   = st.text_input("API Key B",type="password",key="llm_b_key_inp")
        if st.button("Carregar Modelo B",use_container_width=True):
            if key_b.strip():
                try:
                    st.session_state.llm_b = _criar_llm(prov_b,mod_b,key_b.strip(),temp,max_tok)
                    st.success(f"✅ Modelo B: **{mod_b}**")
                except Exception as e: st.error(f"❌ {e}")


def sidebar() -> None:
    usr = usuario_atual()
    st.sidebar.header("⚙️ Administração")
    st.sidebar.caption(f"👤 **{usr['display_name']}** · _{usr['role']}_")

    _painel_metricas()

    modos = ["💬 Chat RAG"]
    if tem_permissao("banco_query"): modos.append("🗄️ Banco de Dados")
    modo = st.sidebar.radio("Módulo",modos,horizontal=True,label_visibility="collapsed")
    st.session_state.modo_tela = "chat" if "Chat" in modo else "banco"
    st.sidebar.divider()

    abas, handlers = [], []
    if tem_permissao("gerenciar_docs"):    abas.append("📄 Docs");     handlers.append(painel_documentos)
    if tem_permissao("configurar_modelo"): abas.append("🤖 Modelo");   handlers.append(_painel_modelo)
    if tem_permissao("banco_query"):       abas.append("🗄️ Banco");    handlers.append(painel_conexao_db)
    if tem_permissao("gerenciar_docs"):    abas.append("👥 Usuários"); handlers.append(painel_usuarios)
    if tem_permissao("ver_audit"):         abas.append("🔎 Audit");    handlers.append(painel_audit_ui)

    if abas:
        tabs = st.sidebar.tabs(abas)
        for tab,handler in zip(tabs,handlers):
            with tab: handler()

    st.sidebar.divider()
    if st.sidebar.button("🚪 Encerrar Sessão",use_container_width=True):
        audit("logout")
        for k,v in _DEFAULTS.items(): st.session_state[k]=v
        st.rerun()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    _startup_run()
    if verificar_timeout(): st.rerun(); return
    if not st.session_state.logged_in:
        pagina_login(); return

    # Carrega histórico persistente uma vez por sessão
    if not st.session_state.historico_carregado and st.session_state.username:
        hist = carregar_historico(st.session_state.username)
        if hist: st.session_state.chat_history = hist
        st.session_state.historico_carregado = True

    sidebar()
    if st.session_state.modo_tela == "banco" and tem_permissao("banco_query"):
        pagina_banco_dados()
    else:
        pagina_chat()


if __name__ == "__main__":
    main()
