"""
config.py — Configuração central via pydantic-settings.
Carrega e valida variáveis de ambiente / .env na inicialização.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key:  Optional[str] = Field(None, description="OpenAI API Key")
    groq_api_key:    Optional[str] = Field(None, description="Groq API Key")

    # ── Sessão / Segurança ────────────────────────────────────────────────
    session_timeout_minutes: int  = Field(60,  ge=0,   description="0 = sem timeout")
    max_requests_per_minute: int  = Field(20,  ge=1,   description="Rate-limit por usuário")
    log_level: str                = Field("INFO")

    # ── RAG ──────────────────────────────────────────────────────────────
    chunk_size:            int   = Field(1000, ge=100)
    chunk_overlap:         int   = Field(150,  ge=0)
    retriever_k:           int   = Field(5,    ge=1,  description="Chunks a recuperar")
    reranker_top_n:        int   = Field(4,    ge=1,  description="Chunks após reranking")
    enable_reranking:      bool  = Field(True,  description="Habilitar CrossEncoder reranking")
    enable_hybrid_search:  bool  = Field(True,  description="Habilitar busca híbrida BM25+semântica")
    enable_llm_cache:      bool  = Field(True,  description="Habilitar cache de respostas LLM")
    llm_cache_ttl_hours:   int   = Field(24,   ge=1)
    enable_auto_summary:   bool  = Field(True,  description="Gerar resumo automático ao indexar docs")
    enable_suggestions:    bool  = Field(True,  description="Sugestões de perguntas relacionadas")

    # ── App ───────────────────────────────────────────────────────────────
    max_upload_size_mb:    int   = Field(200,  ge=1)
    docs_path:             str   = Field("dados_docs")
    faiss_path:            str   = Field("faiss_index")
    cache_dir:             str   = Field(".pleno_cache")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid:
            raise ValueError(f"log_level inválido '{v}'. Use: {valid}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton — instância criada uma única vez por processo."""
    return Settings()
