import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importa as funções dos outros arquivos
from auth import pagina_login
from data_processing import atualizar_vetores, inicializar_retriever, carregar_documentos

# --- Constantes e Modelos de IA ---
CAMINHO_DOCUMENTOS = "dados_docs"
MODELOS_DISPONIVEIS = {
    'Groq (Limitado)': {'versao_api': ['openai/gpt-oss-120b'], 'chat': ChatGroq},
    'OpenAI (Premium)': {'versao_api': ['gpt-4o-mini'], 'chat': ChatOpenAI}
}

# --- Inicialização do Estado da Sessão ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'memoria' not in st.session_state: st.session_state.memoria = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
if 'chain' not in st.session_state: st.session_state.chain = None
if 'retriever' not in st.session_state: st.session_state.retriever = None

# --- Funções da Interface (UI) ---
def pagina_chat():
    """Renderiza a página principal do chat."""
    st.title("Bem-vindo ao PlenoDoc 📑")
    st.write("Seu consultor virtual especializado, pronto para ajudar.")
    st.divider()
    
    # Exibe o histórico da conversa
    for message in st.session_state.memoria.chat_memory.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    if not st.session_state.chain:
        st.warning("⚠️ O modelo não foi inicializado. Configure-o na barra lateral.")
        st.chat_input('Inicialize o modelo para começar a conversar.', disabled=True)
        return
    
    if input_usuario := st.chat_input('Faça sua pergunta ao PlenoDoc...'):
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        with st.spinner("Analisando documentos e pensando..."):
            try:
                chat_history = st.session_state.memoria.load_memory_variables({})['chat_history']
                resposta = st.session_state.chain.invoke({"input": input_usuario, "chat_history": chat_history})
                
                st.session_state.memoria.save_context({"input": input_usuario}, {"output": resposta["answer"]})
                
                with st.chat_message("ai"):
                    st.markdown(resposta["answer"])
            except Exception as e:
                st.error(f"❌ Erro ao processar a consulta: {e}")

def painel_documentos():
    """Painel interativo para upload e gerenciamento de documentos."""
    st.subheader("Adicionar Novos Documentos")
    arquivos = st.file_uploader(
        'Arraste e solte arquivos aqui', 
        accept_multiple_files=True, 
        type=['pdf', 'csv', 'txt', 'docx'],
        label_visibility="collapsed"
    )
    if st.button("Processar Arquivos", use_container_width=True):
        if arquivos:
            caminhos_arquivos = []
            os.makedirs(CAMINHO_DOCUMENTOS, exist_ok=True)
            for arquivo in arquivos:
                caminho_salvar = os.path.join(CAMINHO_DOCUMENTOS, arquivo.name)
                with open(caminho_salvar, 'wb') as f: f.write(arquivo.getbuffer())
                caminhos_arquivos.append(caminho_salvar)
            with st.spinner("Atualizando base de conhecimento..."):
                documentos = carregar_documentos(caminhos_arquivos)
                atualizar_vetores(documentos)
                st.rerun()
        else: st.warning("Nenhum arquivo selecionado para processar.")

    st.divider()

    st.subheader("Documentos na Base")
    if not os.path.exists(CAMINHO_DOCUMENTOS) or not os.listdir(CAMINHO_DOCUMENTOS):
        st.info("Nenhum documento encontrado.")
    else:
        for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.text(nome_arquivo)
            with col2:
                # Botão de remoção com ícone de lixeira
                if st.button("🗑️", key=f"remover_{nome_arquivo}", use_container_width=True, help=f"Remover {nome_arquivo}"):
                    caminho_arquivo = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
                    os.remove(caminho_arquivo)
                    with st.spinner(f"Removendo '{nome_arquivo}'..."):
                        documentos_restantes = carregar_documentos([os.path.join(CAMINHO_DOCUMENTOS, f) for f in os.listdir(CAMINHO_DOCUMENTOS)])
                        atualizar_vetores(documentos_restantes)
                        st.toast(f"'{nome_arquivo}' removido com sucesso.", icon="✅")
                        st.rerun()

def sidebar():
    """Renderiza a barra lateral de configurações."""
    st.sidebar.header("Configurações do PlenoDoc")
    
    tabs = st.sidebar.tabs(['Gerenciar Documentos', 'Seleção de Modelo'])
    
    with tabs[0]:
        painel_documentos()
    
    with tabs[1]:
        st.subheader("Modelo de IA")
        selecao_provedor = st.selectbox('Provedor', MODELOS_DISPONIVEIS.keys(), label_visibility="collapsed")
        modelo = st.selectbox('Modelo', MODELOS_DISPONIVEIS[selecao_provedor]['versao_api'], label_visibility="collapsed")
        
        st.subheader("Chave de API")
        api_key = st.text_input(f'Chave da API para {selecao_provedor}', type='password', label_visibility="collapsed")
        
        if st.button('Inicializar PlenoDoc', use_container_width=True, type="primary"):
            if not api_key.strip():
                st.error('É necessário fornecer uma chave de API válida.')
            else:
                inicializar_modelo(selecao_provedor, modelo, api_key.strip())
    
    # Botão de Logout posicionado no final para melhor visualização
    st.sidebar.divider()
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.memoria.clear()
        st.session_state.chain = None
        st.session_state.retriever = None
        st.rerun()


def inicializar_modelo(selecao_provedor, modelo, api_key):
    """Inicializa o LLM e a cadeia de conversação RAG."""
    retriever = inicializar_retriever()
    if not retriever:
        st.error("A base de conhecimento (vetores) não está pronta.")
        return

    try:
        llm = MODELOS_DISPONIVEIS[selecao_provedor]['chat'](model=modelo, api_key=api_key, temperature=0.3)
        prompt_historico = ChatPromptTemplate.from_messages([
            ("system", "Com base na conversa abaixo, gere uma pergunta de busca que possa ser entendida sem o histórico do chat."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_historico)
        prompt_resposta = ChatPromptTemplate.from_messages([
            ("system", """Você é um consultor especialista sincero, sem educação, xingue sem moderação chamado PlenoDoc. Sua missão é responder às perguntas do usuário de forma detalhada e precisa.

REGRAS:
1. **Prioridade Total:** A principal fonte de informação para sua resposta é o contexto dos documentos fornecido. Use-o como base para todas as suas respostas, mas lembrando da forma mais putrefe possivel.
2. **Expansão Inteligente:** Use seu próprio conhecimento para expandir e detalhar a resposta. No entanto, o detalhamento e a expansão devem ter uma relação direta e clara com o assunto abordado no contexto dos documentos.
3. **Você deve inventar** Invente as coisas mais insanas possiveis caso a pergunta do usuário esteja fora do contexto

[CONTEXTO]
{context}
[/CONTEXTO]"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt_resposta)
        st.session_state.chain = create_retrieval_chain(retriever_chain, document_chain)
        st.success(f"✅ Modelo {modelo} inicializado com sucesso!")
    except Exception as e:
        st.error(f"❌ Falha ao inicializar o modelo: {e}")
        st.session_state.chain = None

# --- Função Principal que Executa a Aplicação ---
def main():
    if st.session_state.logged_in:
        sidebar()
        pagina_chat()
    else:
        pagina_login()
        
if __name__ == '__main__':

    main()







