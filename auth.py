import streamlit as st

USUARIOS = {
    "Administrador": "1234"
}

def pagina_login():
    """Renderiza a interface de autenticação do sistema."""
    st.title("PlenoDoc 📑")
    st.markdown("Autenticação necessária para acessar a base de conhecimento.")
    
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        col1, col2 = st.columns([1, 2])
        with col1:
            login_button = st.form_submit_button("Entrar", use_container_width=True)

    if login_button:
        if username in USUARIOS and USUARIOS[username] == password:
            st.session_state.logged_in = True
            st.success("Acesso autorizado.")
            st.rerun()
        else:
            st.error("Credenciais inválidas.")