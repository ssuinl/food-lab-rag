"""F2003~F2005: 검색어 추출(프롬프트1) → 위키 검색 → 후보 선별(프롬프트2) → 본문 수집 → 청킹 → external_docs 색인

- 항목 자체 문서: 레시피 이름은 항상 검색어에 포함하고, 제목이 레시피 이름과 같은 문서는 선별 결과와 상관없이 색인한다.
- 수집 캐시(NF2002): (레시피, 검색어) 단위로 기록. 이미 수집한 검색어는 다시 검색·수집하지 않는다.
- 같은 문서: 본문은 pageid 단위로 한 번만 받고, 같은 청크 텍스트의 임베딩은 재사용한다.
  청크 ID 는 '항목ID:문서ID:순번' 으로 레시피마다 따로 둔다(명세 F2005: 여러 항목이 같은 문서를 모아도 덮어쓰지 않음).
- PDF 본문은 이용 조건 확인 전이라 색인하지 않는다.
- 오인 매칭으로 판정된 문서(PHASE 1-B ingredient_links 의 REJECTED)와 동음이의 문서는 색인하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .llm import LLMError, ask, parse_json
from .prompts import P1_SYSTEM, P1_USER, P2_SYSTEM, P2_USER
from .settings import LAB, PROJECT
from .wiki import LICENSE, WikiError

CACHE = PROJECT / "cache"
STATUS_LABEL = {
    "ITEM_SELF_DOC_FOUND": "레시피와 같은 이름의 외부 문서를 찾음 (같은 음식의 일반 설명)",
    "NO_ITEM_SELF_DOC": "레시피 자체를 다룬 외부 문서 없음 — 레시피 메타데이터와 원문 링크만 항목 근거",
}


def _norm(s: str) -> str:
    return re.sub(r"[\s·\-,()（）\[\]&]", "", s or "").lower()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(p)


def blocked_titles() -> set[str]:
    """PHASE 1-B 에서 오인 매칭으로 판정된 위키 문서 제목 (없으면 빈 집합)."""
    p = LAB / "data_check" / "derived" / "ingredient_links.json"
    d = _read(p, {})
    out = set()
    for x in d.get("links", []):
        if x.get("relation") in ("REJECTED", "AMBIGUOUS") and x.get("wiki_title"):
            out.add(x["wiki_title"])
    return out


def item_status(seq: str) -> dict:
    return _read(CACHE / "items" / f"{seq}.json", {"rcp_seq": seq, "queries": {}, "docs": {}, "rejected": [], "item_self": None})


def main_ingredients(recipe: dict, k: int = 6) -> list[str]:
    out = []
    for i in recipe["ingredients"]:
        if i.get("section") in ("SEASONING", "BROTH"): continue
        b = i.get("base") or i.get("name")
        if b and b not in out: out.append(b)
    return out[:k]


def extract_queries(chat, recipe, question, history_text) -> tuple[list[str], str]:
    main = ", ".join(main_ingredients(recipe)) or "정보 없음"
    try:
        out = parse_json(ask(chat, P1_SYSTEM, P1_USER.format(name=recipe["name"], category=recipe["category"], method=recipe["method"],
                                                            main=main, history=history_text or "(없음)", question=question)), list)
    except LLMError:
        out = None
    if out:
        qs = [str(q).strip() for q in out if str(q).strip()][:4]
        how = "prompt1"
    else:
        qs = main_ingredients(recipe, 2); how = "fallback(주재료)"
    # F2003: 항목 자체를 다룬 문서는 항상 검색 대상
    allq = [recipe["name"]] + [q for q in qs if _norm(q) != _norm(recipe["name"])]
    return list(dict.fromkeys(allq)), how


def select_candidates(chat, recipe, question, cands, max_docs) -> tuple[list[dict], str]:
    if not cands: return [], "no_candidates"
    lines = "\n".join(f"- {c['title']}: {c['snippet'][:120]}" for c in cands)
    try:
        out = parse_json(ask(chat, P2_SYSTEM.format(max_docs=max_docs), P2_USER.format(
            name=recipe["name"], category=recipe["category"], main=", ".join(main_ingredients(recipe)),
            question=question, candidates=lines)), list)
    except LLMError:
        return [], "prompt2_failed"
    if out is None: return [], "prompt2_unparsable"
    titles = {c["title"]: c for c in cands}
    picked = []
    for x in out:
        if not isinstance(x, dict): continue
        t = str(x.get("title", "")).strip()
        rel = x.get("relation") if x.get("relation") in ("FOOD_BACKGROUND", "INGREDIENT_BACKGROUND", "COOKING_METHOD") else "FOOD_BACKGROUND"
        if t in titles and t not in {p["title"] for p in picked}:
            picked.append({**titles[t], "relation": rel, "reason": str(x.get("reason", ""))[:120], "selected_by": "prompt2"})
    return picked[:max_docs], "prompt2"


def chunk_text(cfg, text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"],
                                        separators=["\n\n", "\n", ". ", " ", ""])
    return [c for c in sp.split_text(text) if c.strip()]


def collect(cfg, recipe, question, history_text, chat, emb, wiki, ext_col, progress=lambda m: None) -> dict:
    seq = recipe["id"]
    st = item_status(seq)
    res = {"rcp_seq": seq, "queries": [], "query_method": None, "new_queries": [], "skipped_queries": [],
           "selected": [], "rejected": [], "indexed_chunks": 0, "reused_vectors": 0, "errors": [], "item_self": st.get("item_self")}
    # NF2002: 같은 레시피·같은 질문(·같은 대화 맥락)이면 프롬프트1 결과를 재사용 → 같은 검색어 → 재수집 0회
    qk = CACHE / "queries" / f"{_h(seq + '|' + _norm(question) + '|' + _norm(history_text or ''))}.json"
    qc = _read(qk, None)
    if qc and qc.get("queries"):
        progress("같은 질문의 검색어를 재사용합니다 (LLM 호출 없음)")
        queries, how = qc["queries"], qc["method"] + "(cached)"
    else:
        progress("검색어를 고르는 중 (프롬프트 1)")
        queries, how = extract_queries(chat, recipe, question, history_text)
        if how == "prompt1":
            _write(qk, {"rcp_seq": seq, "question": question, "queries": queries, "method": how,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    res["queries"], res["query_method"] = queries, how
    new_q = [q for q in queries if _norm(q) not in st["queries"]]
    res["skipped_queries"] = [q for q in queries if _norm(q) in st["queries"]]
    res["new_queries"] = new_q
    if not new_q:
        progress("이미 수집한 검색어입니다 — 다시 수집하지 않음"); return res
    blocked = blocked_titles()
    cands, seen, failed_q = [], set(), set()
    for q in new_q:
        progress(f"위키백과 검색: {q}")
        ck = CACHE / "wiki_search" / f"{_h(q)}.json"
        hit = _read(ck, None)
        if hit is None:
            try:
                hit = {"query": q, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": wiki.search(q, cfg["wiki_search_limit"])}
                _write(ck, hit)
            except WikiError as e:
                res["errors"].append(f"검색 실패({q}): {e}"); failed_q.add(q); continue
        for c in hit["results"]:
            if c["title"] in seen: continue
            seen.add(c["title"])
            if c["title"] in blocked:
                res["rejected"].append({"title": c["title"], "reason": "오인 매칭 이력(PHASE 1-B)"}); continue
            cands.append({**c, "query": q})
    # 항목 자체 문서: 제목이 레시피 이름과 같은 후보는 무조건 포함
    forced = [{**c, "relation": "FOOD_BACKGROUND", "reason": "레시피 이름과 같은 제목", "selected_by": "item_self_rule"}
              for c in cands if _norm(c["title"]) == _norm(recipe["name"])]
    progress("후보 문서를 고르는 중 (프롬프트 2)")
    picked, how2 = select_candidates(chat, recipe, question, [c for c in cands if c not in forced], cfg["wiki_max_docs_per_question"])
    if how2 != "prompt2": res["errors"].append(f"후보 선별 단계: {how2}")
    chosen = forced + [p for p in picked if p["title"] not in {f["title"] for f in forced}]
    for c in cands:
        if c["title"] not in {x["title"] for x in chosen}:
            res["rejected"].append({"title": c["title"], "reason": "선별 프롬프트에서 제외"})
    for doc in chosen:
        if doc["title"] in st["docs"]:
            continue
        progress(f"본문 수집: {doc['title']}")
        pk = CACHE / "wiki_pages" / f"{doc['page_id']}.json"
        page = _read(pk, None)
        if page is None:
            try:
                page = wiki.fetch(doc["title"])
            except WikiError as e:
                res["errors"].append(f"본문 실패({doc['title']}): {e}"); continue
            if page is None:
                res["errors"].append(f"문서 없음({doc['title']})"); continue
            page["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); _write(pk, page)
        if page.get("disambiguation") or not page.get("text", "").strip():
            res["rejected"].append({"title": doc["title"], "reason": "동음이의 문서 또는 본문 없음"}); continue
        chunks = chunk_text(cfg, page["text"])
        ids = [f"{seq}:wiki-{page['page_id']}:{n:03d}" for n in range(len(chunks))]
        have = set(ext_col.get(ids=ids)["ids"]) if ids else set()
        todo = [(i, c) for i, c in zip(ids, chunks) if i not in have]
        vecs = {}
        for i, c in todo:  # 다른 레시피가 이미 임베딩한 같은 청크는 벡터 재사용
            hh = _h(c)
            g = ext_col.get(where={"chunk_hash": hh}, limit=1, include=["embeddings"])
            if g["ids"]:
                vecs[i] = list(g["embeddings"][0]); res["reused_vectors"] += 1
        need = [(i, c) for i, c in todo if i not in vecs]
        try:
            for k in range(0, len(need), 32):
                part = need[k:k + 32]
                for (i, _), v in zip(part, emb.embed([c for _, c in part])): vecs[i] = v
        except Exception as e:
            res["errors"].append(f"임베딩 실패({doc['title']}): {type(e).__name__}"); continue
        if todo:
            ext_col.upsert(ids=[i for i, _ in todo], documents=[c for _, c in todo], embeddings=[vecs[i] for i, _ in todo],
                           metadatas=[{"rcp_seq": seq, "doc_id": f"wiki-{page['page_id']}", "doc_title": page["title"],
                                       "source_url": page.get("url") or "", "source_type": "wikipedia",
                                       "source_label": "한국어 위키백과", "license": LICENSE, "relation": doc["relation"],
                                       "selected_by": doc["selected_by"], "query": doc.get("query", ""),
                                       "chunk_index": int(i.rsplit(":", 1)[1]), "chunk_hash": _h(c),
                                       "revision": str(page.get("revision") or ""), "collected_at": page.get("fetched_at", "")}
                                      for i, c in todo])
        res["indexed_chunks"] += len(todo)
        st["docs"][doc["title"]] = {"url": page.get("url"), "page_id": page["page_id"], "relation": doc["relation"],
                                    "selected_by": doc["selected_by"], "reason": doc.get("reason", ""), "chunks": len(chunks)}
        res["selected"].append(st["docs"][doc["title"]] | {"title": doc["title"]})
    doc_failed = any(e.startswith(("본문 실패", "임베딩 실패")) for e in res["errors"])
    for q in new_q:   # 실패한 검색어(또는 문서 수집이 실패한 회차)는 다음 질문에서 다시 시도할 수 있게 기록하지 않는다
        if q not in failed_q and not doc_failed:
            st["queries"][_norm(q)] = {"query": q, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if forced and any(f["title"] in st["docs"] for f in forced):
        st["item_self"] = "ITEM_SELF_DOC_FOUND"
    elif st.get("item_self") is None:
        st["item_self"] = "NO_ITEM_SELF_DOC"
    st["rejected"] = (st.get("rejected", []) + res["rejected"])[-50:]
    _write(CACHE / "items" / f"{seq}.json", st)
    res["item_self"] = st["item_self"]
    return res
