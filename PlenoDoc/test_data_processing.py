"""
test_data_processing.py — Testes para chunking, deduplicação e metadados RAG.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# ── Deduplicação ─────────────────────────────────────────────────────────────

def test_deduplicar_remove_identicos():
    from data_processing import _deduplicar
    docs = [
        Document(page_content="conteúdo A", metadata={}),
        Document(page_content="conteúdo A", metadata={}),   # duplicata
        Document(page_content="conteúdo B", metadata={}),
    ]
    unique, removidos = _deduplicar(docs)
    assert len(unique) == 2
    assert removidos   == 1


def test_deduplicar_sem_duplicatas():
    from data_processing import _deduplicar
    docs = [Document(page_content=f"texto {i}", metadata={}) for i in range(5)]
    unique, removidos = _deduplicar(docs)
    assert len(unique) == 5
    assert removidos   == 0


def test_deduplicar_lista_vazia():
    from data_processing import _deduplicar
    unique, removidos = _deduplicar([])
    assert unique   == []
    assert removidos == 0


def test_deduplicar_preserva_metadados():
    from data_processing import _deduplicar
    docs = [
        Document(page_content="texto único", metadata={"source": "arquivo.pdf", "page": 1}),
    ]
    unique, _ = _deduplicar(docs)
    assert unique[0].metadata["source"] == "arquivo.pdf"
    assert unique[0].metadata["page"]   == 1


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_fracionar_cria_chunks():
    from data_processing import _fracionar
    texto_longo = "palavra " * 500       # ~4000 chars > chunk_size=1000
    docs  = [Document(page_content=texto_longo, metadata={"source": "test.txt"})]
    chunks = _fracionar(docs)
    assert len(chunks) > 1


def test_fracionar_preserva_source():
    from data_processing import _fracionar
    docs   = [Document(page_content="texto curto", metadata={"source": "origem.pdf"})]
    chunks = _fracionar(docs)
    assert all(c.metadata.get("source") == "origem.pdf" for c in chunks)


def test_fracionar_nao_ultrapassa_chunk_size():
    from data_processing import _fracionar
    from config import get_settings
    cfg  = get_settings()
    docs = [Document(page_content="a " * 2000, metadata={})]
    for chunk in _fracionar(docs):
        assert len(chunk.page_content) <= cfg.chunk_size + 50  # margem para overlap


# ── Carregamento de documentos ────────────────────────────────────────────────

def test_carregar_formato_invalido():
    from data_processing import carregar_documentos
    _, erros = carregar_documentos(["/fake/arquivo.xyz"])
    assert len(erros) == 1
    assert "não suportado" in erros[0].lower()


def test_carregar_arquivo_inexistente():
    from data_processing import carregar_documentos
    _, erros = carregar_documentos(["/caminho/que/nao/existe.pdf"])
    assert len(erros) == 1


def test_carregar_txt_normaliza_source(tmp_path):
    from data_processing import carregar_documentos
    arq = tmp_path / "doc.txt"
    arq.write_text("conteúdo de teste", encoding="utf-8")
    docs, erros = carregar_documentos([str(arq)])
    assert len(erros)  == 0
    assert len(docs)   >= 1
    assert docs[0].metadata["source"]     == "doc.txt"
    assert docs[0].metadata["tipo_fonte"] == "arquivo"


# ── Resumos ───────────────────────────────────────────────────────────────────

def test_gerar_resumo_sem_llm(streamlit_session_state):
    from data_processing import gerar_resumo_documento
    streamlit_session_state["llm"] = None
    resultado = gerar_resumo_documento("doc.pdf", [])
    assert resultado == ""


def test_gerar_resumo_com_llm(streamlit_session_state, tmp_path, monkeypatch):
    import data_processing
    monkeypatch.setattr(data_processing, "_SUMMARIES_FILE", str(tmp_path / "summaries.json"))
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Resumo gerado pelo LLM.")
    streamlit_session_state["llm"] = mock_llm
    chunks = [Document(page_content="conteúdo relevante", metadata={})]
    resultado = data_processing.gerar_resumo_documento("manual.pdf", chunks)
    assert resultado == "Resumo gerado pelo LLM."


# ── Cache LLM ─────────────────────────────────────────────────────────────────

def test_cache_miss(tmp_cache):
    from cache_llm import get_cached
    assert get_cached("pergunta nova", "gpt-4o") is None


def test_cache_set_e_get(tmp_cache):
    from cache_llm import get_cached, set_cached
    set_cached("minha pergunta", "gpt-4o", "minha resposta")
    resultado = get_cached("minha pergunta", "gpt-4o")
    assert resultado == "minha resposta"


def test_cache_key_diferente_por_modelo(tmp_cache):
    from cache_llm import get_cached, set_cached
    set_cached("pergunta X", "gpt-4o",   "resposta A")
    set_cached("pergunta X", "llama-70b", "resposta B")
    assert get_cached("pergunta X", "gpt-4o")   == "resposta A"
    assert get_cached("pergunta X", "llama-70b") == "resposta B"


# ── Memória persistente ───────────────────────────────────────────────────────

def test_salvar_e_carregar_historico(tmp_cache):
    from memory import salvar_mensagem, carregar_historico
    from langchain_core.messages import HumanMessage, AIMessage
    salvar_mensagem("user1", "human", "Olá!")
    salvar_mensagem("user1", "ai",    "Olá, como posso ajudar?")
    hist = carregar_historico("user1")
    assert len(hist) == 2
    assert isinstance(hist[0], HumanMessage)
    assert isinstance(hist[1], AIMessage)
    assert hist[0].content == "Olá!"


def test_limpar_historico(tmp_cache):
    from memory import salvar_mensagem, carregar_historico, limpar_historico
    salvar_mensagem("user2", "human", "mensagem")
    limpar_historico("user2")
    assert carregar_historico("user2") == []


def test_historico_isolado_por_usuario(tmp_cache):
    from memory import salvar_mensagem, carregar_historico
    salvar_mensagem("userA", "human", "mensagem A")
    salvar_mensagem("userB", "human", "mensagem B")
    hist_a = carregar_historico("userA")
    hist_b = carregar_historico("userB")
    assert len(hist_a) == 1
    assert len(hist_b) == 1
    assert hist_a[0].content == "mensagem A"
    assert hist_b[0].content == "mensagem B"
