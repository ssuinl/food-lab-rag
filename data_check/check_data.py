"""Food Lab 1차 데이터 검증 — 식품안전나라 조리식품 레시피 DB(COOKRCP01) + 위키백과 매칭

실행 (streamlit_lab 폴더에서):
    .venv\\Scripts\\python data_check\\check_data.py

하는 일
  1. .env 에서 식품안전나라 키를 읽는다 (키 값은 절대 출력하지 않는다)
  2. COOKRCP01 전체를 1,000건 단위로 받아 원본 그대로 data_check/raw/ 에 저장한다 (수정·보완 없음)
  3. 건수 · 결측 · 중복 · 재료 형식 · 조리법 유무를 검사한다
  4. 메뉴명을 한국어 위키백과 문서 제목과 대조해 매칭률과 오인 매칭 후보를 기록한다
  5. 결과를 화면에 출력하고 data_check/report.json 에 저장한다

기존 파일은 읽기만 하고 바꾸지 않는다.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import dotenv_values, load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)

load_dotenv(LAB / ".env")

KEY_CANDIDATES = ["FOODSAFETY_KEY", "FOOD_SAFETY_KEY", "FOODSAFETY_API_KEY", "FOOD_API_KEY",
                  "COOKRCP_KEY", "RECIPE_KEY", "RECIPE_API_KEY", "MFDS_KEY", "FOODSAFETYKOREA_KEY"]
KEY_NAME = next((n for n in KEY_CANDIDATES if os.getenv(n)), None)
KEY = os.getenv(KEY_NAME) if KEY_NAME else None

BASE = "http://openapi.foodsafetykorea.go.kr/api/{key}/COOKRCP01/json/{start}/{end}"
WIKI_API = "https://ko.wikipedia.org/w/api.php"
CONTACT = os.getenv("WIKI_CONTACT", "").strip()
UA = f"FoodLab-Check/0.1 ({CONTACT}) python-requests" if CONTACT else "FoodLab-Check/0.1 python-requests"

STEP_FIELDS = [f"MANUAL{i:02d}" for i in range(1, 21)]
CHECK_FIELDS = ["RCP_SEQ", "RCP_NM", "RCP_PARTS_DTLS", "RCP_PAT2", "RCP_WAY2",
                "INFO_ENG", "HASH_TAG", "ATT_FILE_NO_MAIN", "RCP_NA_TIP"]


def safe(msg: str) -> str:
    """오류 문구에 키가 섞여 나오지 않게 가린다."""
    return str(msg).replace(KEY, "***") if KEY else str(msg)


def title(text):
    print("\n" + "=" * 64 + f"\n{text}\n" + "=" * 64)


def fetch_range(start: int, end: int) -> dict:
    url = BASE.format(key=KEY, start=start, end=end)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"   [재시도 {attempt + 1}/3] {start}~{end}: {type(e).__name__}: {safe(e)[:120]}")
            time.sleep(3)
    raise SystemExit(f"{start}~{end} 구간을 받지 못했습니다. 네트워크나 키를 확인하세요.")


def collect() -> tuple[list[dict], int, dict]:
    first = fetch_range(1, 1000)
    body = first.get("COOKRCP01", {})
    result = body.get("RESULT", {})
    if result.get("CODE") not in ("INFO-000", None):
        raise SystemExit(f"API 오류: {result.get('CODE')} {result.get('MSG')}")
    total = int(body.get("total_count", 0))
    (RAW / "cookrcp01_1_1000.json").write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
    rows = list(body.get("row", []))
    print(f"   1~1000 → {len(rows)}건 (total_count={total})")
    start = 1001
    while start <= total:
        end = min(start + 999, total)
        time.sleep(1)
        data = fetch_range(start, end)
        (RAW / f"cookrcp01_{start}_{end}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        part = data.get("COOKRCP01", {}).get("row", [])
        print(f"   {start}~{end} → {len(part)}건")
        rows.extend(part)
        start = end + 1
    return rows, total, result


def blank(v) -> bool:
    return v is None or str(v).strip() == ""


def quality(rows: list[dict]) -> dict:
    n = len(rows)
    missing = {f: sum(blank(r.get(f)) for r in rows) for f in CHECK_FIELDS}
    steps = [sum(not blank(r.get(f)) for f in STEP_FIELDS) for r in rows]
    seq_dup = [k for k, c in Counter(r.get("RCP_SEQ") for r in rows).items() if c > 1]
    name_cnt = Counter((r.get("RCP_NM") or "").strip() for r in rows)
    name_dup = {k: c for k, c in name_cnt.items() if c > 1 and k}
    parts = [r.get("RCP_PARTS_DTLS") or "" for r in rows]
    fmt = {
        "줄바꿈 포함": sum("\n" in p for p in parts),
        "쉼표 포함": sum("," in p for p in parts),
        "괄호/분량 표기(g, 개, 큰술 등) 포함": sum(any(u in p for u in ["g", "개", "큰술", "작은술", "컵", "ml"]) for p in parts),
        "'재료' 머리말 포함": sum("재료" in p for p in parts),
        "평균 길이(자)": round(sum(len(p) for p in parts) / n, 1) if n else 0,
    }
    return {
        "rows": n,
        "unique_RCP_SEQ": len({r.get("RCP_SEQ") for r in rows}),
        "unique_RCP_NM": len([k for k in name_cnt if k]),
        "missing_by_field": missing,
        "no_steps": sum(s == 0 for s in steps),
        "steps_min_max_avg": [min(steps, default=0), max(steps, default=0),
                              round(sum(steps) / n, 1) if n else 0],
        "dup_RCP_SEQ": seq_dup[:20],
        "dup_RCP_NM": dict(list(name_dup.items())[:20]),
        "dup_RCP_NM_count": len(name_dup),
        "RCP_PAT2_dist": dict(Counter(r.get("RCP_PAT2") for r in rows).most_common()),
        "RCP_WAY2_dist": dict(Counter(r.get("RCP_WAY2") for r in rows).most_common()),
        "parts_format": fmt,
        "parts_samples": [{"RCP_SEQ": r.get("RCP_SEQ"), "RCP_NM": r.get("RCP_NM"),
                           "RCP_PARTS_DTLS": (r.get("RCP_PARTS_DTLS") or "")[:200]}
                          for r in rows[:3] + rows[len(rows) // 2: len(rows) // 2 + 2]],
    }


def wiki_titles(titles: list[str]) -> dict:
    """제목 → {'final': 최종 제목 or None, 'disambig': bool}. 50개씩 묶어 조회."""
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        for attempt in range(4):
            r = requests.get(WIKI_API, params={
                "action": "query", "titles": "|".join(batch), "redirects": 1,
                "prop": "pageprops", "ppprop": "disambiguation",
                "format": "json", "formatversion": 2}, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After") or 5))
                continue
            r.raise_for_status()
            break
        q = r.json().get("query", {})
        mapping = {t: t for t in batch}
        for n in q.get("normalized", []):
            for k, v in mapping.items():
                if v == n["from"]:
                    mapping[k] = n["to"]
        redir = {d["from"]: d["to"] for d in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for orig, t in mapping.items():
            t2 = redir.get(t, t)
            p = pages.get(t2)
            if p and not p.get("missing") and not p.get("invalid"):
                out[orig] = {"final": t2, "redirected": t2 != t,
                             "disambig": "disambiguation" in (p.get("pageprops") or {})}
            else:
                out[orig] = {"final": None, "redirected": False, "disambig": False}
        time.sleep(0.3)
    return out


def wiki_match(rows: list[dict]) -> dict:
    names = sorted({(r.get("RCP_NM") or "").strip() for r in rows if not blank(r.get("RCP_NM"))})
    nospace = {n: n.replace(" ", "") for n in names}
    last = {n: n.split()[-1] for n in names}
    cands = sorted(set(names) | set(nospace.values()) | set(last.values()))
    print(f"   메뉴명 {len(names)}개, 조회할 제목 후보 {len(cands)}개")
    res = wiki_titles(cands)
    exact, generic, none_, disambig = [], [], [], []
    for n in names:
        hit = res.get(n, {}).get("final") and res[n] or (res.get(nospace[n], {}).get("final") and res[nospace[n]])
        if hit:
            exact.append((n, hit["final"]))
            if hit["disambig"]:
                disambig.append((n, hit["final"]))
            continue
        g = res.get(last[n], {})
        if g.get("final") and last[n] != n:
            generic.append((n, g["final"]))
            if g["disambig"]:
                disambig.append((n, g["final"]))
        else:
            none_.append(n)
    total = len(names) or 1
    return {
        "menu_names": len(names),
        "exact_title_match": len(exact),
        "exact_rate": round(len(exact) / total * 100, 1),
        "generic_last_word_match": len(generic),
        "generic_rate": round(len(generic) / total * 100, 1),
        "no_match": len(none_),
        "no_match_rate": round(len(none_) / total * 100, 1),
        "disambiguation_hits": len(disambig),
        "examples_exact": exact[:15],
        "examples_generic": generic[:25],
        "examples_none": none_[:25],
        "examples_disambig": disambig[:15],
    }


def main():
    title("1. 키 확인")
    if not KEY:
        names = [k for k in dotenv_values(LAB / ".env").keys()]
        print("식품안전나라 키를 찾지 못했습니다. .env 에 들어 있는 변수 '이름'(값 아님):")
        for k in names:
            print("   -", k)
        print("위 이름 중 식품안전나라 키를 FOODSAFETY_KEY 로 바꾸거나, 그 이름을 알려 주세요.")
        sys.exit(1)
    print(f"{KEY_NAME} 를 찾았습니다. (길이 {len(KEY)}) — 값은 출력하지 않습니다")
    print(f"위키 User-Agent 연락처: {'있음' if CONTACT else '없음 (분당 10회 제한, 느릴 수 있음)'}")

    title("2. 레시피 수집 (원본 그대로 저장)")
    rows, total, result = collect()

    title("3. 품질 검사")
    q = quality(rows)
    print(json.dumps({k: v for k, v in q.items() if k != "parts_samples"}, ensure_ascii=False, indent=2))
    print("\n재료정보 샘플:")
    for s in q["parts_samples"]:
        print(f"   [{s['RCP_SEQ']}] {s['RCP_NM']}: {s['RCP_PARTS_DTLS']!r}")

    title("4. 위키백과 매칭")
    try:
        w = wiki_match(rows)
        print(json.dumps(w, ensure_ascii=False, indent=2))
    except Exception as e:
        w = {"error": f"{type(e).__name__}: {e}"}
        print("위키 매칭 실패:", w["error"])

    report = {"api_result": result, "total_count": total, "collected": len(rows),
              "quality": q, "wiki": w, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    title("끝 — data_check/report.json 저장됨. 이 화면 출력이나 report.json 을 확인하세요")


if __name__ == "__main__":
    main()
