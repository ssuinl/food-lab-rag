"""ChromaDB 컬렉션 접근. 임베딩은 우리가 계산해서 넣는다(모델 혼용 방지)."""
from __future__ import annotations

import chromadb

from .settings import chroma_path


class StoreError(RuntimeError):
    pass


def client(cfg):
    p = chroma_path(cfg)
    p.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(p))


def get_collection(cfg, name: str, model: str | None, create: bool = False):
    c = client(cfg)
    existing = {col.name for col in c.list_collections()}
    if name not in existing:
        if not create:
            return None
        return c.create_collection(name=name, metadata={"hnsw:space": "cosine", "embedding_model": model or ""})
    col = c.get_collection(name=name)
    stored = (col.metadata or {}).get("embedding_model")
    if model and stored and stored != model:
        raise StoreError(f"컬렉션 '{name}'은 '{stored}' 모델로 만들어졌습니다. 현재 설정 '{model}'과 섞을 수 없습니다. "
                         f"--rebuild 로 다시 만드세요.")
    return col


def drop_collection(cfg, name: str):
    c = client(cfg)
    if name in {col.name for col in c.list_collections()}:
        c.delete_collection(name)
