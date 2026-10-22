"""F2010 도메인 특화(냉장고 맞춤 조리 가이드) · F2011 자연어 조건 탐색."""
from __future__ import annotations

import re
import statistics

from .llm import LLMError, ask, parse_json
from .prompts import FRIDGE_QUESTION, NL_JUDGE_SYSTEM, NL_JUDGE_USER, NL_PARSE_SYSTEM, NL_PARSE_USER

CATEGORIES = ["반찬", "일품", "후식", "밥", "국&찌개", "기타"]
_MEDIANS: dict = {}


def fridge_question(owned: list[str]) -> str:
    return FRIDGE_QUESTION.format(owned=", ".join(owned) if owned else "(입력 없음)")


def _medians(recipe_col) -> dict:
    """'적은/낮은' 같은 상대 조건의 기준값: 영양 이상치가 없는 레시피의 중앙값."""
    key = id(recipe_col)
    if key not in _MEDIANS:
        mds = recipe_col.get(include=["metadatas"])["metadatas"]
        ok = [m for m in mds if not m.get("nutrition_suspect")]
        out = {}
        for k in ("sodium_mg", "kcal", "protein_g"):
            vals = [m[k] for m in ok if isinstance(m.get(k), (int, float))]
            out[k] = statistics.median(vals) if vals else None
        _MEDIANS[key] = out
    return _MEDIANS[key]


def _num(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def parse_condition(chat, condition: str) -> dict | None:
    try:
        d = parse_json(ask(chat, NL_PARSE_SYSTEM, NL_PARSE_USER.format(condition=condition)), dict)
    except LLMError:
        return None
    if d is None:
        return None
    cat = d.get("category") if d.get("category") in CATEGORIES else None
    lst = lambda k: [str(x).strip() for x in (d.get(k) or []) if str(x).strip()][:6]
    unsup = lst("unsupported")
    # 판단 불가 조건(조리 시간 등)이 판단 조건에 중복으로 들어오면 뺀다
    NOINFO = re.compile(r"\d+\s*분|시간|가격|비용|난이도|쉬운|간단")
    def overlaps(j): return bool(NOINFO.search(j)) or any(set(re.findall(r"\w{2,}", j)) & set(re.findall(r"\w{2,}", u)) for u in unsup)
    judge = [j for j in lst("judge") if not overlaps(j)]
    return {"category": cat, "low_sodium": bool(d.get("low_sodium")), "low_kcal": bool(d.get("low_kcal")),
            "high_protein": bool(d.get("high_protein")), "max_sodium_mg": _num(d.get("max_sodium_mg")),
            "max_kcal": _num(d.get("max_kcal")), "include": lst("include"),
            # 제외 재료는 실제 재료 이름만 (예: '맵게 하는 재료' 같은 설명문은 버림)
            "exclude": [x for x in lst("exclude") if " " not in x and len(x) <= 6],
            "judge": judge, "unsupported": unsup,
            "search_text": str(d.get("search_text") or condition)[:200]}


def nl_search(cfg, recipe_col, emb, chat, condition: str, n_candidates: int = 20) -> dict:
    """1) LLM 은 문장을 구조화만 한다 (요리 종류·영양·재료·기타·판단 불가 조건)
    2) 요리 종류·영양 수치·재료 조건은 코드로 강제 적용 (ChromaDB 메타데이터 필터)
    3) 남은 후보를 의미 유사도 순으로 보여 준다. 결과 이유는 원본 메타데이터 값으로만 만든다(LLM 문장 아님).
    4) 데이터에 없는 조건(조리 시간 등)은 판단하지 않고 화면에 알린다."""
    p = parse_condition(chat, condition)
    if p is None:
        return {"ok": False, "error": "조건을 해석하지 못했습니다. 조금 더 구체적으로 적어 보세요.", "candidates": 0,
                "results": [], "applied": [], "unsupported": [], "notes": []}
    med = _medians(recipe_col)
    clauses, applied = [], []
    if p["category"]:
        clauses.append({"category": p["category"]}); applied.append(f"요리 종류 = {p['category']}")
    num_rules = []   # (키, 상한, 설명)
    if p["max_sodium_mg"]: num_rules.append(("sodium_mg", p["max_sodium_mg"], f"나트륨 ≤ {p['max_sodium_mg']:g}mg"))
    elif p["low_sodium"] and med.get("sodium_mg"): num_rules.append(("sodium_mg", med["sodium_mg"], f"나트륨 ≤ {med['sodium_mg']:g}mg (전체 중앙값)"))
    if p["max_kcal"]: num_rules.append(("kcal", p["max_kcal"], f"열량 ≤ {p['max_kcal']:g}kcal"))
    elif p["low_kcal"] and med.get("kcal"): num_rules.append(("kcal", med["kcal"], f"열량 ≤ {med['kcal']:g}kcal (전체 중앙값)"))
    for k, v, label in num_rules:
        clauses.append({k: {"$lte": v}}); applied.append(label)
    if p["high_protein"] and med.get("protein_g"):
        clauses.append({"protein_g": {"$gte": med["protein_g"]}}); applied.append(f"단백질 ≥ {med['protein_g']:g}g (전체 중앙값)")
    if num_rules or p["high_protein"]:
        clauses.append({"nutrition_suspect": False}); applied.append("영양 수치 이상치 레시피 제외")
    def run(cl):
        where = None if not cl else cl[0] if len(cl) == 1 else {"$and": cl}
        try:
            r = recipe_col.query(query_embeddings=[qv], n_results=n_candidates * 3, where=where, include=["metadatas", "documents"])
            out = []
            for m, d in zip(r["metadatas"][0], r["documents"][0]):
                m = dict(m); m["_doc"] = d; out.append(m)
            return out
        except Exception:
            return []

    def has(m, ing): return ing in (m.get("main_ingredients", "") + "," + m.get("name", ""))

    def has_any(m, ing):
        """제외 재료는 양념까지 포함한 재료 원문 전체에서 찾는다 (예: 후추·고춧가루는 주재료 목록에 없음)."""
        full = m.get("main_ingredients", "") + "," + m.get("name", "")
        for ln in (m.get("_doc") or "").splitlines():
            if ln.startswith(("재료 원문:", "양념·육수 재료:", "고명·곁들임:")): full += "," + ln
        return ing in full

    def ing_filter(cs):
        if p["include"]: cs = [m for m in cs if all(has(m, i) for i in p["include"])]
        if p["exclude"]: cs = [m for m in cs if not any(has_any(m, i) for i in p["exclude"])]
        return cs

    # 의미 검색 문장: 해석된 검색문 + 판단 조건(맵지 않은 등)은 검색 순위에만 반영
    qv = emb.embed([" ".join([p["search_text"]] + p["judge"])])[0]
    if p["include"]: applied.append("포함 재료: " + ", ".join(p["include"]))
    if p["exclude"]: applied.append("제외 재료: " + ", ".join(p["exclude"]))
    notes = []
    cands = ing_filter(run(clauses))
    if not cands and len(clauses) > (1 if p["category"] else 0):
        # 수치 조건 때문에 0건이면 요리 종류만 남기고 다시 찾고, 그 사실을 알린다
        cands = ing_filter(run(clauses[:1] if p["category"] else []))
        if cands: notes.append("영양 수치 조건을 만족하는 레시피가 없어 수치 조건을 빼고 찾았습니다.")
    def reason(m):
        bits = [m.get("category", "")]
        if isinstance(m.get("sodium_mg"), (int, float)): bits.append(f"나트륨 {m['sodium_mg']:g}mg")
        if isinstance(m.get("kcal"), (int, float)): bits.append(f"열량 {m['kcal']:g}kcal")
        if isinstance(m.get("protein_g"), (int, float)) and p["high_protein"]: bits.append(f"단백질 {m['protein_g']:g}g")
        hit = [i for i in p["include"] if has(m, i)]
        if hit: bits.append("포함: " + ", ".join(hit))
        if m.get("nutrition_suspect"): bits.append("영양 수치 검토 필요")
        return " · ".join(b for b in bits if b) + " (원본 데이터 값)"

    cands = cands[:n_candidates]
    if not p["judge"]:
        results = [{"rcp_seq": m["rcp_seq"], "meta": m, "reason": reason(m)} for m in cands[:6]]
        return {"ok": True, "error": None, "results": results, "candidates": len(cands), "applied": applied,
                "unsupported": p["unsupported"], "notes": notes, "parsed": p}

    # 판단 조건(예: 맵지 않은, 아이 간식)은 확실한 조건을 통과한 후보 안에서만 LLM 이 판단한다.
    # 판단 불가 조건(조리 시간 등)은 이미 분리돼 있어 여기 들어오지 않는다. 후보 밖 번호는 버린다.
    def doc_line(m, head):
        for ln in (m.get("_doc") or "").splitlines():
            if ln.startswith(head): return ln[len(head):].strip()[:220]
        return ""
    lines = [f"- {m['rcp_seq']} | {m['display_name']} | {m['category']} | {m['method']} | 재료: {doc_line(m, '재료 원문:')} | 해시태그: {doc_line(m, '해시태그:')}"
             for m in cands]
    try:
        out = parse_json(ask(chat, NL_JUDGE_SYSTEM, NL_JUDGE_USER.format(judge="\n".join(p["judge"]), candidates="\n".join(lines))), list)
    except LLMError:
        out = None
    if out is None:
        return {"ok": False, "error": "판단 조건('" + "', '".join(p["judge"]) + "')을 확인하는 단계에서 응답을 받지 못했습니다. 다시 시도해 주세요.",
                "results": [], "candidates": len(cands), "applied": applied, "unsupported": p["unsupported"], "notes": notes, "parsed": p}
    by = {m["rcp_seq"]: m for m in cands}
    results, seen = [], set()
    for x in out:
        if not isinstance(x, dict): continue
        sq = str(x.get("rcp_seq", "")).strip()
        if sq in by and sq not in seen:
            seen.add(sq)
            why = str(x.get("reason", "")).strip()[:80]
            results.append({"rcp_seq": sq, "meta": by[sq], "reason": reason(by[sq]) + (f" · AI 판단: {why}" if why else "")})
    applied.append("AI 판단 조건: " + ", ".join(p["judge"]))
    return {"ok": True, "error": None, "results": results[:6], "candidates": len(cands), "applied": applied,
            "unsupported": p["unsupported"], "notes": notes, "parsed": p}
