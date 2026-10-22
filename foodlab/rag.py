"""F2006~F2008: LangGraph retrieve → generate.

retrieve: 선택 레시피의 메타데이터 청크는 검색 없이 항상 포함(F2006) + external_docs 에서 그 레시피로 범위를 좁혀 의미 유사도 검색
          → 병합, 중복 청크 제거(ID·본문 해시), 근거 번호 부여
generate: 프롬프트3 — 근거만 사용, [근거 N] 표기, 근거 없으면 '확인되지 않음'(F2007)
후처리:  답변이 실제로 인용한 근거만 남김, 인용이 없으면 그 사실을 표시(F2008)
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLMError, ask
from .prompts import P3_SYSTEM, P3_USER, P3G_SYSTEM, P3G_USER

CITE = re.compile(r"\[근거\s*([\d,\s]+)\]")


class RAGState(TypedDict, total=False):
    recipe: dict
    question: str
    history: str
    fridge: str
    sources: list
    answer: str
    cited: list
    notes: list


def _h(s): return hashlib.sha256(s.encode()).hexdigest()[:20]


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1; nb = math.sqrt(sum(x * x for x in b)) or 1
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def mmr(query_vec, cands, k, lam=0.5):
    """F2014(선택): 후보를 넉넉히 받은 뒤 다양성을 고려해 k개 선택."""
    chosen = []
    pool = list(cands)
    while pool and len(chosen) < k:
        best = max(pool, key=lambda c: lam * _cos(query_vec, c["vec"]) - (1 - lam) * max([_cos(c["vec"], s["vec"]) for s in chosen] or [0]))
        chosen.append(best); pool.remove(best)
    return chosen


def build_graph(cfg, recipe_col, ext_col, emb, chat):
    retrieve = build_retrieve(cfg, recipe_col, ext_col, emb)
    generate = make_generate(chat, P3_SYSTEM, lambda st, ctx: P3_USER.format(context=ctx, history=st.get("history") or "(없음)", question=st["question"]))
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve"); g.add_edge("retrieve", "generate"); g.add_edge("generate", END)
    return g.compile()


def build_retrieve(cfg, recipe_col, ext_col, emb):
    """레시피 대화용 retrieve 노드 (검증 도구에서 생성 없이 근거 목록만 비교할 때도 쓴다)."""
    top_k = int(cfg["top_k_external"])

    def retrieve(state: RAGState) -> RAGState:
        r, q = state["recipe"], state["question"]
        notes, sources, seen_ids, seen_txt = [], [], set(), set()
        meta = recipe_col.get(ids=[f"recipe:{r['id']}"], include=["documents", "metadatas"])
        if meta["ids"]:
            m = meta["metadatas"][0]
            sources.append({"id": meta["ids"][0], "type": "recipe_meta", "label": "레시피 메타데이터",
                            "title": m.get("display_name") or r["name"], "url": m.get("source_url"),
                            "text": meta["documents"][0], "nutrition_suspect": bool(m.get("nutrition_suspect"))})
            seen_ids.add(meta["ids"][0]); seen_txt.add(_h(meta["documents"][0]))
        else:
            notes.append("NO_META_INDEX")
        if ext_col is not None and ext_col.count() > 0:
            try:
                qv = emb.embed([f"{r['name']} {q}"])[0]
                n = top_k * 3 if cfg.get("mmr") else top_k
                res = ext_col.query(query_embeddings=[qv], n_results=n, where={"rcp_seq": r["id"]},
                                    include=["documents", "metadatas", "distances", "embeddings"] if cfg.get("mmr") else ["documents", "metadatas", "distances"])
                cands = []
                for i, (cid, doc, md) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0])):
                    c = {"id": cid, "doc": doc, "md": md, "dist": res["distances"][0][i]}
                    if cfg.get("mmr"): c["vec"] = list(res["embeddings"][0][i])
                    cands.append(c)
                if cfg.get("mmr"): cands = mmr(qv, cands, top_k)
                for c in cands:
                    th = _h(c["doc"])
                    if c["id"] in seen_ids or th in seen_txt: continue   # F2006 중복 제거
                    seen_ids.add(c["id"]); seen_txt.add(th)
                    sources.append({"id": c["id"], "type": "wikipedia", "label": "위키백과", "title": c["md"].get("doc_title"),
                                    "url": c["md"].get("source_url"), "text": c["doc"], "relation": c["md"].get("relation"),
                                    "distance": round(c["dist"], 3)})
            except Exception as e:
                notes.append(f"EXTERNAL_SEARCH_FAILED:{type(e).__name__}")
        else:
            notes.append("NO_EXTERNAL_DOCS")
        for n, s in enumerate(sources, 1): s["n"] = n
        return {"sources": sources, "notes": notes}

    return retrieve


def _ctx(srcs):
    return "\n\n".join(
        f"[근거 {s['n']}] ({s['label']}: {s['title']})" + (" [주의: 영양 수치 이상치 있음]" if s.get("nutrition_suspect") else "") + f"\n{s['text']}"
        for s in srcs)


def make_generate(chat, system, user_fn):
    """generate 노드: 근거만으로 답변 → 실제로 인용된 근거만 남김(F2008)."""
    def generate(state: RAGState) -> RAGState:
        srcs = state.get("sources", [])
        if not srcs:
            return {"answer": "제공된 문서에서 확인되지 않습니다. (검색된 근거가 없습니다)", "cited": [], "notes": state.get("notes", []) + ["NO_SOURCES"]}
        try:
            ans = ask(chat, system, user_fn(state, _ctx(srcs)))
        except LLMError:
            return {"answer": "답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.", "cited": [], "notes": state.get("notes", []) + ["LLM_FAILED"]}
        nums = {int(x) for g in CITE.findall(ans) for x in g.split(",") if x.strip().isdigit()}
        cited = [s for s in srcs if s["n"] in nums]
        return {"answer": ans, "cited": cited, "notes": state.get("notes", []) + ([] if cited else ["NO_CITATION"])}
    return generate


def build_general_graph(cfg, recipe_col, ext_col, emb, chat, n_recipes: int = 5):
    """레시피를 고르지 않은 질문: retrieve(레시피 메타데이터 Top-n + 수집된 외부 문서 Top-k) → generate."""
    top_k = int(cfg["top_k_external"])

    def retrieve(state: RAGState) -> RAGState:
        q = state["question"]; fridge = state.get("fridge") or ""
        notes, sources, seen_txt = [], [], set()
        qv = emb.embed([(f"냉장고 재료: {fridge}. " if fridge else "") + q])[0]
        r = recipe_col.query(query_embeddings=[qv], n_results=n_recipes, include=["documents", "metadatas", "distances"])
        for cid, doc, md, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            seen_txt.add(_h(doc))
            sources.append({"id": cid, "type": "recipe_meta", "label": "레시피 메타데이터", "rcp_seq": md.get("rcp_seq"),
                            "title": md.get("display_name") or md.get("name"), "url": md.get("source_url"), "text": doc,
                            "nutrition_suspect": bool(md.get("nutrition_suspect")), "distance": round(d, 3)})
        if ext_col is not None and ext_col.count() > 0:
            try:
                e = ext_col.query(query_embeddings=[qv], n_results=top_k * 3, include=["documents", "metadatas", "distances"])
                k = 0
                for cid, doc, md, d in zip(e["ids"][0], e["documents"][0], e["metadatas"][0], e["distances"][0]):
                    th = _h(doc)
                    if th in seen_txt: continue      # 같은 위키 청크가 여러 레시피에 색인돼 있어도 한 번만
                    seen_txt.add(th); k += 1
                    sources.append({"id": cid, "type": "wikipedia", "label": "위키백과", "title": md.get("doc_title"),
                                    "url": md.get("source_url"), "text": doc, "relation": md.get("relation"), "distance": round(d, 3)})
                    if k >= top_k: break
            except Exception as ex:
                notes.append(f"EXTERNAL_SEARCH_FAILED:{type(ex).__name__}")
        else:
            notes.append("NO_EXTERNAL_DOCS")
        for n, s_ in enumerate(sources, 1): s_["n"] = n
        return {"sources": sources, "notes": notes}

    generate = make_generate(chat, P3G_SYSTEM, lambda st, ctx: P3G_USER.format(
        context=ctx, fridge=st.get("fridge") or "(입력 없음)", history=st.get("history") or "(없음)", question=st["question"]))
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve); g.add_node("generate", generate)
    g.add_edge(START, "retrieve"); g.add_edge("retrieve", "generate"); g.add_edge("generate", END)
    return g.compile()


def history_text(msgs: list[dict], n: int = 4) -> str:
    tail = msgs[-n:]
    return "\n".join(f"{'사용자' if m['role'] == 'user' else 'AI 셰프'}: {m['content'][:300]}" for m in tail)
