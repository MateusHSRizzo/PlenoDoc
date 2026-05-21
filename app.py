import streamlit as st
import os

# Configuração da página DEVE ser a primeira instrução Streamlit
st.set_page_config(page_title="PlenoDoc", page_icon="📑", layout="wide")

from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from auth import pagina_login
from data_processing import adicionar_ao_indice, reconstruir_indice, inicializar_retriever

CAMINHO_DOCUMENTOS = "dados_docs"
MODELOS_DISPONIVEIS = {
    'Groq (Fast)': {'versao_api': ['llama3-8b-8192', 'llama3-70b-8192'], 'chat': ChatGroq},
    'OpenAI (Premium)': {'versao_api': ['gpt-4o-mini', 'gpt-3.5-turbo'], 'chat': ChatOpenAI}
}

# Inicialização de Estado
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'memoria' not in st.session_state: st.session_state.memoria = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
if 'chain' not in st.session_state: st.session_state.chain = None

def pagina_chat():
    st.title("PlenoDoc 📑")
    st.write("Módulo de Consulta Automatizada de Suporte Técnico.")
    st.divider()
    
    for message in st.session_state.memoria.chat_memory.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    if not st.session_state.chain:
        st.warning("⚠️ Instância LLM offline. Configure a API Key na barra lateral.")
        st.chat_input('Aguardando inicialização do modelo...', disabled=True)
        return
    
    if input_usuario := st.chat_input('Descreva o problema ou pesquise o procedimento...'):
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        with st.spinner("Buscando referências na base..."):
            try:
                chat_history = st.session_state.memoria.load_memory_variables({})['chat_history']
                resposta = st.session_state.chain.invoke({"input": input_usuario, "chat_history": chat_history})
                
                st.session_state.memoria.save_context({"input": input_usuario}, {"output": resposta["answer"]})
                
                with st.chat_message("ai"):
                    st.markdown(resposta["answer"])
            except Exception as e:
                st.error(f"❌ Erro de processamento: {e}")

def painel_documentos():
    st.subheader("Gestão de Base (RAG)")
    arquivos = st.file_uploader(
        'Upload de Procedimentos', 
        accept_multiple_files=True, 
        type=['pdf', 'csv', 'txt', 'docx'],
        label_visibility="collapsed"
    )
    if st.button("Processar e Indexar", use_container_width=True):
        if arquivos:
            caminhos_novos = []
            os.makedirs(CAMINHO_DOCUMENTOS, exist_ok=True)
            for arquivo in arquivos:
                caminho_salvar = os.path.join(CAMINHO_DOCUMENTOS, arquivo.name)
                with open(caminho_salvar, 'wb') as f: f.write(arquivo.getbuffer())
                caminhos_novos.append(caminho_salvar)
            
            with st.spinner("Vetorizando documentos..."):
                sucesso, msg = adicionar_ao_indice(caminhos_novos)
                if sucesso:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else: st.warning("Selecione arquivos antes de processar.")

    st.divider()
    st.subheader("Repositório Local")
    
    if not os.path.exists(CAMINHO_DOCUMENTOS) or not os.listdir(CAMINHO_DOCUMENTOS):
        st.info("Repositório vazio.")
    else:
        for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.text(nome_arquivo)
            with col2:
                if st.button("🗑️", key=f"remover_{nome_arquivo}"):
                    os.remove(os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo))
                    with st.spinner("Reconstruindo FAISS..."):
                        arquivos_restantes = os.listdir(CAMINHO_DOCUMENTOS)
                        caminhos_restantes = [os.path.join(CAMINHO_DOCUMENTOS, f) for f in arquivos_restantes]
                        
                        sucesso, msg = reconstruir_indice(caminhos_restantes)
                        if sucesso:
                            st.toast(f"Arquivo removido da base.", icon="✅")
                            # Reseta a chain para forçar nova leitura do retriever na próxima inicialização
                            st.session_state.chain = None 
                            st.rerun()
                        else:
                            st.error(msg)

def inicializar_modelo(selecao_provedor, modelo, api_key):
    retriever = inicializar_retriever()
    if not retriever:
        st.error("Base de vetores indisponível. Processe os documentos primeiro.")
        return

    try:
        llm = MODELOS_DISPONIVEIS[selecao_provedor]['chat'](model=modelo, api_key=api_key, temperature=0.1)
        
        prompt_historico = ChatPromptTemplate.from_messages([
            ("system", "Considerando o histórico, reformule a pergunta do usuário para que seja compreendida de forma independente, focando na busca de documentos."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_historico)
        
        prompt_resposta = ChatPromptTemplate.from_messages([
            ("system", """Você é o PlenoDoc, assistente especialista de suporte técnico.
Sua única fonte de verdade é o contexto técnico fornecido abaixo. 

REGRAS ESTRITAS:
1. Responda APENAS com base no contexto.
2. Seja objetivo, pragmático e direto nas resoluções.
3. Se a informação não existir no contexto, responda: "Informação não encontrada na base de conhecimento documentada."
4. NUNCA invente comandos, IPs, senhas ou procedimentos de infraestrutura.

[CONTEXTO_DOCUMENTAL]
{context}
[/CONTEXTO_DOCUMENTAL]"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, prompt_resposta)
        st.session_state.chain = create_retrieval_chain(retriever_chain, document_chain)
        st.success(f"✅ Instância conectada: {modelo}")
    except Exception as e:
        st.error(f"❌ Erro de conexão com a API: {e}")
        st.session_state.chain = None

def sidebar():
    st.sidebar.header("Administração")
    tabs = st.sidebar.tabs(['Documentos', 'Configuração LLM'])
    
    with tabs[0]:
        painel_documentos()
    
    with tabs[1]:
        selecao_provedor = st.selectbox('Provedor', MODELOS_DISPONIVEIS.keys())
        modelo = st.selectbox('Modelo', MODELOS_DISPONIVEIS[selecao_provedor]['versao_api'])
        api_key = st.text_input(f'API Key', type='password')
        
        if st.button('Carregar Modelo', type="primary", use_container_width=True):
            if not api_key.strip():
                st.error('Forneça a credencial da API.')
            else:
                inicializar_modelo(selecao_provedor, modelo, api_key.strip())
    
    st.sidebar.divider()
    if st.sidebar.button("Encerrar Sessão", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.memoria.clear()
        st.session_state.chain = None
        st.rerun()

def main():
    if st.session_state.logged_in:
        sidebar()
        pagina_chat()
    else:
        pagina_login()
        
if __name__ == '__main__':
    main()