"""
app.py — Ponto de entrada do PlenoDoc.

Melhorias aplicadas:
  - Logging estruturado em arquivo + console
  - .env carregado automaticamente
  - Streaming de resposta do LLM (token a token)
  - Citação de fontes RAG após cada resposta
  - Botões de feedback 👍 / 👎 por mensagem
  - Export do histórico do chat em Markdown
  - Temperatura e max_tokens configuráveis via slider
  - Botão "🗄️ Só Banco" (inicia LLM sem precisar de docs indexados)
  - URL indexável direto pelo painel de documentos
  - Interface adaptada ao perfil do usuário (admin / operator / viewer)
  - Painel de usuários acessível para admins
"""

import os
import logging
import datetime
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Carrega .env antes de qualquer outra coisa
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — arquivo rotacionado + console
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    if root.handlers:           # evita duplicar handlers em reruns do Streamlit
        return
    root.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/plenodoc.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

_setup_logging()
logger = logging.getLogger("plenodoc.app")

# ---------------------------------------------------------------------------
# st.set_page_config — obrigatoriamente antes de qualquer outro st.*
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PlenoDoc",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports do projeto (após set_page_config)
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from auth import (
    pagina_login, verificar_timeout, tem_permissao,
    usuario_atual, painel_usuarios,
)
from data_processing import (
    adicionar_ao_indice, reconstruir_indice,
    inicializar_retriever, adicionar_url_ao_indice,
)
from database import painel_conexao_db, pagina_banco_dados, inicializar_estado_db

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CAMINHO_DOCUMENTOS = "dados_docs"

MODELOS_DISPONIVEIS = {
    "Groq (Fast)": {
        "versao_api": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "chat": ChatGroq,
    },
    "OpenAI (Premium)": {
        "versao_api": ["gpt-4o-mini", "gpt-4o"],
        "chat": ChatOpenAI,
    },
}

# ---------------------------------------------------------------------------
# Inicialização do estado da sessão
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "logged_in":      False,
    "username":       "",
    "user_role":      "viewer",
    "display_name":   "",
    "last_activity":  0.0,
    "chat_history":   [],        # [HumanMessage | AIMessage]
    "chat_feedback":  {},        # {índice: "up" | "down"}
    "chat_sources":   {},        # {índice_ai: [nomes_de_arquivo]}
    "chain":          None,
    "retriever":      None,
    "llm":            None,
    "modo_tela":      "chat",
    "llm_temperatura": 0.1,
    "llm_max_tokens":  1000,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

inicializar_estado_db()


# ===========================================================================
# CHAT RAG
# ===========================================================================

def _exportar_chat_markdown() -> bytes:
    """Serializa o histórico do chat em Markdown para download."""
    agora  = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    linhas = [f"# PlenoDoc — Histórico do Chat\n_Exportado em {agora}_\n\n---\n"]
    for i, msg in enumerate(st.session_state.chat_history):
        role = "👤 **Usuário**" if isinstance(msg, HumanMessage) else "🤖 **PlenoDoc**"
        linhas.append(f"{role}:\n{msg.content}\n")
        fb = st.session_state.chat_feedback.get(i)
        if fb:
            linhas.append(f"_Avaliação: {'👍 Útil' if fb == 'up' else '👎 Não útil'}_\n")
        sources = st.session_state.chat_sources.get(i, [])
        if sources:
            linhas.append(f"_Fontes: {', '.join(sources)}_\n")
        linhas.append("---\n")
    return "\n".join(linhas).encode("utf-8")


def pagina_chat() -> None:
    st.title("PlenoDoc 📑")
    st.caption("Módulo de Consulta Automatizada de Suporte Técnico.")
    st.divider()

    # Barra de ações acima do histórico
    if st.session_state.chat_history:
        c1, c2, _ = st.columns([0.18, 0.18, 0.64])
        if c1.button("🗑️ Limpar chat", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.chat_feedback = {}
            st.session_state.chat_sources  = {}
            st.rerun()
        c2.download_button(
            "⬇️ Exportar",
            data=_exportar_chat_markdown(),
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Exibição do histórico
    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

            # Fontes (somente mensagens do assistente)
            if role == "assistant":
                sources = st.session_state.chat_sources.get(i, [])
                if sources:
                    with st.expander(f"📎 Fontes ({len(sources)})", expanded=False):
                        for src in sorted(sources):
                            st.markdown(f"• `{src}`")

                # Botões de feedback
                fb      = st.session_state.chat_feedback.get(i)
                fc1, fc2, _ = st.columns([0.045, 0.045, 0.91])
                if fc1.button(
                    "👍", key=f"up_{i}",
                    help="Resposta útil",
                    type="primary" if fb == "up" else "secondary",
                ):
                    st.session_state.chat_feedback[i] = "up"
                    st.rerun()
                if fc2.button(
                    "👎", key=f"dw_{i}",
                    help="Resposta não útil",
                    type="primary" if fb == "down" else "secondary",
                ):
                    st.session_state.chat_feedback[i] = "down"
                    st.rerun()

    # Input desabilitado se chain não está pronta
    if not st.session_state.chain:
        st.info("ℹ️ Instância LLM offline. Configure a API Key na barra lateral.")
        st.chat_input("Aguardando inicialização do modelo…", disabled=True)
        return

    if prompt := st.chat_input("Descreva o problema ou pesquise o procedimento…"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Streaming ──────────────────────────────────────────────────────
        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_answer   = ""
            context_docs  = []

            try:
                for chunk in st.session_state.chain.stream({
                    "input":        prompt,
                    "chat_history": st.session_state.chat_history,
                }):
                    if "context" in chunk:
                        context_docs = chunk["context"]
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
                        placeholder.markdown(full_answer + "▌")

                placeholder.markdown(full_answer)

                # Salva no histórico
                user_idx = len(st.session_state.chat_history)
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                ai_idx = len(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=full_answer))

                # Fontes
                fontes: set[str] = set()
                for doc in context_docs:
                    src = doc.metadata.get("source", "")
                    if src:
                        fontes.add(os.path.basename(str(src)))
                if fontes:
                    st.session_state.chat_sources[ai_idx] = sorted(fontes)
                    with st.expander(f"📎 Fontes ({len(fontes)})", expanded=False):
                        for f in sorted(fontes):
                            st.markdown(f"• `{f}`")

                logger.info("Resposta gerada para '%s…' | fontes: %s", prompt[:40], fontes)

            except Exception as e:
                placeholder.error(f"❌ Erro de processamento: {e}")
                logger.error("Erro no chat: %s", e)


# ===========================================================================
# PAINEL DE DOCUMENTOS
# ===========================================================================

def painel_documentos() -> None:
    st.subheader("Gestão de Base (RAG)")

    # Upload de arquivos
    arquivos = st.file_uploader(
        "Arquivos",
        accept_multiple_files=True,
        type=["pdf", "csv", "txt", "docx"],
        label_visibility="collapsed",
    )
    if st.button("⬆️ Processar Arquivos", use_container_width=True, disabled=not bool(arquivos)):
        os.makedirs(CAMINHO_DOCUMENTOS, exist_ok=True)
        caminhos = []
        for arq in arquivos:
            dest = os.path.join(CAMINHO_DOCUMENTOS, arq.name)
            with open(dest, "wb") as f:
                f.write(arq.getbuffer())
            caminhos.append(dest)
        with st.spinner("Vetorizando…"):
            ok, msg = adicionar_ao_indice(caminhos)
        if ok:
            st.session_state.chain = None
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    # Indexar URL
    st.divider()
    st.subheader("🌐 Indexar URL")
    url_input = st.text_input("URL", placeholder="https://docs.exemplo.com/pagina", label_visibility="collapsed")
    if st.button("⬆️ Carregar URL", use_container_width=True, disabled=not bool(url_input and url_input.startswith("http"))):
        with st.spinner("Carregando e indexando URL…"):
            ok, msg = adicionar_url_ao_indice(url_input.strip())
        if ok:
            st.session_state.chain = None
            st.success(msg)
        else:
            st.error(msg)

    # Repositório local
    st.divider()
    st.subheader("📋 Repositório Local")
    docs = []
    if os.path.exists(CAMINHO_DOCUMENTOS):
        docs = sorted(f for f in os.listdir(CAMINHO_DOCUMENTOS) if not f.startswith("."))

    if not docs:
        st.info("Repositório vazio.")
        return

    for nome in docs:
        c1, c2 = st.columns([0.85, 0.15])
        c1.text(nome)
        if c2.button("🗑️", key=f"del_{nome}", use_container_width=True, help=f"Remover {nome}"):
            os.remove(os.path.join(CAMINHO_DOCUMENTOS, nome))
            restantes = [
                os.path.join(CAMINHO_DOCUMENTOS, f)
                for f in os.listdir(CAMINHO_DOCUMENTOS) if not f.startswith(".")
            ]
            with st.spinner("Reconstruindo índice…"):
                ok, msg = reconstruir_indice(restantes)
            if ok:
                st.session_state.chain = None
                st.toast("✅ Arquivo removido e índice reconstruído.")
                st.rerun()
            else:
                st.error(msg)


# ===========================================================================
# INICIALIZAÇÃO DO MODELO
# ===========================================================================

def _criar_llm(provedor: str, modelo: str, api_key: str, temperatura: float, max_tokens: int):
    return MODELOS_DISPONIVEIS[provedor]["chat"](
        model=modelo,
        api_key=api_key,
        temperature=temperatura,
        max_tokens=max_tokens,
    )


def inicializar_modelo(provedor: str, modelo: str, api_key: str, temperatura: float, max_tokens: int) -> None:
    """Inicializa LLM + cadeia RAG completa."""
    retriever = inicializar_retriever()
    if not retriever:
        st.error("⚠️ Base de vetores indisponível. Processe documentos primeiro.")
        return
    try:
        llm = _criar_llm(provedor, modelo, api_key, temperatura, max_tokens)
        st.session_state.llm = llm

        prompt_ctx = ChatPromptTemplate.from_messages([
            ("system", "Reformule a pergunta do usuário de forma independente do histórico, para otimizar a busca nos documentos. Não responda — apenas reformule se necessário."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_ctx)

        prompt_resp = ChatPromptTemplate.from_messages([
            ("system",
             "Você é o PlenoDoc, assistente especialista de suporte técnico.\n"
             "Responda APENAS com base no contexto documental abaixo.\n"
             "Seja objetivo e direto. Se não encontrar no contexto, diga: "
             "\"Informação não encontrada na base de conhecimento.\"\n"
             "NUNCA invente comandos, IPs, senhas ou procedimentos.\n"
             "Quando possível, cite de qual documento a informação veio.\n\n"
             "[CONTEXTO]\n{context}\n[/CONTEXTO]"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        doc_chain = create_stuff_documents_chain(llm, prompt_resp)
        st.session_state.chain = create_retrieval_chain(retriever_chain, doc_chain)
        logger.info("Modelo RAG inicializado: %s / %s", provedor, modelo)
        st.success(f"✅ Modelo **{modelo}** inicializado com RAG.")

    except Exception as e:
        logger.error("Falha ao inicializar modelo: %s", e)
        st.error(f"❌ Falha ao inicializar: {e}")
        st.session_state.chain = None
        st.session_state.llm   = None


def inicializar_llm_standalone(provedor: str, modelo: str, api_key: str, temperatura: float, max_tokens: int) -> None:
    """Inicializa apenas o LLM (sem RAG), para uso no módulo de banco de dados."""
    try:
        st.session_state.llm = _criar_llm(provedor, modelo, api_key, temperatura, max_tokens)
        logger.info("LLM standalone inicializado: %s / %s", provedor, modelo)
        st.success(f"✅ LLM **{modelo}** pronto para consultas ao banco.")
    except Exception as e:
        logger.error("Falha ao inicializar LLM standalone: %s", e)
        st.error(f"❌ Erro: {e}")
        st.session_state.llm = None


# ===========================================================================
# BARRA LATERAL
# ===========================================================================

def sidebar() -> None:
    usr = usuario_atual()
    st.sidebar.header("⚙️ Administração")
    st.sidebar.caption(f"👤 **{usr['display_name']}** · _{usr['role']}_")

    # Navegação principal
    opcoes_nav = ["💬 Chat RAG"]
    if tem_permissao("banco_query"):
        opcoes_nav.append("🗄️ Banco de Dados")

    modo = st.sidebar.radio("Módulo", opcoes_nav, horizontal=True, label_visibility="collapsed")
    st.session_state.modo_tela = "chat" if "Chat" in modo else "banco"
    st.sidebar.divider()

    # Monta abas conforme permissões
    abas_nomes = []
    if tem_permissao("gerenciar_docs"):     abas_nomes.append("📄 Documentos")
    if tem_permissao("configurar_modelo"):  abas_nomes.append("🤖 Modelo")
    if tem_permissao("banco_query"):        abas_nomes.append("🗄️ Banco")
    if tem_permissao("gerenciar_docs"):     abas_nomes.append("👥 Usuários")

    if not abas_nomes:
        st.sidebar.info("Sem permissões de configuração.")
    else:
        abas = st.sidebar.tabs(abas_nomes)
        idx  = 0

        if tem_permissao("gerenciar_docs"):
            with abas[idx]:
                painel_documentos()
            idx += 1

        if tem_permissao("configurar_modelo"):
            with abas[idx]:
                _painel_modelo()
            idx += 1

        if tem_permissao("banco_query"):
            with abas[idx]:
                painel_conexao_db()
            idx += 1

        if tem_permissao("gerenciar_docs"):
            with abas[idx]:
                painel_usuarios()
            idx += 1

    st.sidebar.divider()
    if st.sidebar.button("🚪 Encerrar Sessão", use_container_width=True):
        for k in list(_DEFAULTS.keys()):
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()


def _painel_modelo() -> None:
    """Conteúdo da aba de configuração do modelo LLM."""
    provedor = st.selectbox("Provedor", MODELOS_DISPONIVEIS.keys(), label_visibility="collapsed")
    modelo   = st.selectbox("Modelo",   MODELOS_DISPONIVEIS[provedor]["versao_api"], label_visibility="collapsed")
    api_key  = st.text_input("API Key", type="password", label_visibility="collapsed",
                             placeholder=f"Chave para {provedor}…")

    st.subheader("Parâmetros")
    temp = st.slider(
        "Temperatura",
        min_value=0.0, max_value=1.0,
        value=st.session_state.llm_temperatura,
        step=0.05,
        help="0 = determinístico  |  1 = criativo",
    )
    st.session_state.llm_temperatura = temp

    max_tok = st.slider(
        "Max Tokens (resposta)",
        min_value=256, max_value=4096,
        value=st.session_state.llm_max_tokens,
        step=128,
    )
    st.session_state.llm_max_tokens = max_tok

    c1, c2 = st.columns(2)
    if c1.button("🚀 Chat RAG", type="primary", use_container_width=True,
                 help="Inicializa LLM + RAG (requer documentos indexados)"):
        if not api_key.strip():
            st.error("Informe a API Key.")
        else:
            inicializar_modelo(provedor, modelo, api_key.strip(), temp, max_tok)

    if c2.button("🗄️ Só Banco", use_container_width=True,
                 help="Inicializa LLM apenas para o módulo de banco de dados"):
        if not api_key.strip():
            st.error("Informe a API Key.")
        else:
            inicializar_llm_standalone(provedor, modelo, api_key.strip(), temp, max_tok)

    # Status
    if st.session_state.get("llm"):
        st.success("LLM ativo ✅", icon="🤖")
    else:
        st.info("LLM não inicializado")

    if st.session_state.get("chain"):
        st.success("RAG ativo ✅", icon="📚")


# ===========================================================================
# PONTO DE ENTRADA
# ===========================================================================

def main() -> None:
    if verificar_timeout():
        st.rerun()

    if not st.session_state.logged_in:
        pagina_login()
        return

    sidebar()

    if st.session_state.modo_tela == "banco" and tem_permissao("banco_query"):
        pagina_banco_dados()
    else:
        pagina_chat()


if __name__ == "__main__":
    main()
