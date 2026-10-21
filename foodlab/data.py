"""원본 COOKRCP01 로더 (읽기 전용). 가짜 레코드를 만들지 않는다.

v0.2 대비: 품질 플래그를 원본에서 직접 계산(derived 파일 없이도 경고 표시), 원본 해시 기록,
http 이미지도 허용(도메인은 식품안전나라로 제한), 영양 값 숫자화.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .parse_parts_v1 import parse

DETAIL_BASE = ("https://www.foodsafetykorea.go.kr/portal/cookrcp/cookrcpInfoDtl.do"
               "?menu_no=3608&menu_grp=MENU_NEW03&rcp_seq=")
EXPECTED_ROWS = 1156
# PHASE 1-B 에서 근거와 함께 확인한 재등록 의심 묶음 (병합하지 않고 표시만)
NEAR_DUP = {"788": "919", "919": "788", "807": "1034", "1034": "807", "2980": "2981", "2981": "2980",
            "224": "774", "774": "224", "225": "776", "776": "225"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_raw_rows(raw_dir: Path) -> tuple[list[dict], dict]:
    files = sorted(raw_dir.glob("cookrcp01_*.json"))
    rows, hashes = [], {}
    for f in files:
        hashes[f.name] = sha256(f)
        obj = json.loads(f.read_text(encoding="utf-8-sig"))
        rows.extend(obj["COOKRCP01"]["row"])
    return rows, hashes


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def quality_flags(r: dict) -> list[str]:
    steps = [r.get(f"MANUAL{i:02d}", "") or "" for i in range(1, 21)]
    filled = [i for i, s in enumerate(steps) if s.strip()]
    flags = []
    if not filled: flags.append("STEPS_EMPTY")
    elif filled != list(range(len(filled))): flags.append("STEPS_NUMBER_GAP")
    if not (r.get("RCP_PARTS_DTLS") or "").strip(): flags.append("PARTS_EMPTY")
    e, c, p, f, na = (_num(r.get(k)) or 0 for k in ("INFO_ENG", "INFO_CAR", "INFO_PRO", "INFO_FAT", "INFO_NA"))
    calc = 4 * c + 4 * p + 9 * f
    if e == 0 and calc == 0: flags.append("NUTRITION_ALL_ZERO")
    elif e > 0 and calc / e > 2: flags.append("KCAL_MACRO_MISMATCH_HIGH")
    elif e > 0 and calc / e < 0.5: flags.append("KCAL_MACRO_MISMATCH_LOW")
    if max(c, p, f) > 300: flags.append("MACRO_IMPLAUSIBLE_GT300G")
    if na > 2000 and e < 100: flags.append("SODIUM_IMPLAUSIBLE")
    return flags


NUTRITION_FLAGS = {"NUTRITION_ALL_ZERO", "KCAL_MACRO_MISMATCH_HIGH", "KCAL_MACRO_MISMATCH_LOW",
                   "MACRO_IMPLAUSIBLE_GT300G", "SODIUM_IMPLAUSIBLE"}


def clean_text(v) -> str:
    """표시·색인용 파생 텍스트: <br> → 줄바꿈. 원본 값은 바꾸지 않는다."""
    return re.sub(r"<br\s*/?>", "\n", str(v or ""), flags=re.I).strip()


def clean_step(v) -> str:
    return re.sub(r"^\s*(?:\d{1,2}[.)．、]\s*)+", "", clean_text(v)).strip()


def normalize_rows(rows: list[dict]) -> list[dict]:
    out, seen = [], set()
    names = {}
    for r in rows:
        names.setdefault(str(r.get("RCP_NM", "")).strip(), []).append(str(r.get("RCP_SEQ", "")).strip())
    for r in rows:
        if not isinstance(r, dict): continue
        seq, name = str(r.get("RCP_SEQ", "")).strip(), str(r.get("RCP_NM", "")).strip()
        if not seq or not name or seq in seen: continue
        seen.add(seq)
        parts = str(r.get("RCP_PARTS_DTLS") or "")
        ings = parse(name, parts)["items"] if parts.strip() else []
        steps = [clean_step(r.get(f"MANUAL{j:02d}")) for j in range(1, 21) if str(r.get(f"MANUAL{j:02d}") or "").strip()]
        same = [x for x in names.get(name, []) if x != seq]
        out.append({
            "id": seq, "name": name,
            "display_name": f"{name} (#{seq})" if same else name,
            "category": str(r.get("RCP_PAT2") or "미분류"), "method": str(r.get("RCP_WAY2") or "미분류"),
            "ingredients": ings, "ingredient_text": clean_text(parts), "steps": steps,
            "tip": clean_text(r.get("RCP_NA_TIP")), "hashtag": str(r.get("HASH_TAG") or "").strip(),
            "nutrition": {"kcal": _num(r.get("INFO_ENG")), "carb_g": _num(r.get("INFO_CAR")), "protein_g": _num(r.get("INFO_PRO")),
                          "fat_g": _num(r.get("INFO_FAT")), "sodium_mg": _num(r.get("INFO_NA")), "weight_g": _num(r.get("INFO_WGT"))},
            "image": str(r.get("ATT_FILE_NO_MAIN") or "").strip(),
            "flags": quality_flags(r), "same_name": same, "near_duplicate_of": NEAR_DUP.get(seq),
            "source": DETAIL_BASE + seq,
        })
    return out


def load_recipes(raw_dir: Path) -> tuple[list[dict], dict]:
    rows, hashes = load_raw_rows(raw_dir)
    return normalize_rows(rows), hashes


def safe_image_url(value) -> str:
    """식품안전나라 도메인 이미지 주소만 표시 (http 포함). 가져오거나 검증하지는 않는다."""
    try:
        u = urlparse(str(value).strip())
    except ValueError:
        return ""
    if u.scheme in ("http", "https") and u.netloc.lower() in {"www.foodsafetykorea.go.kr", "foodsafetykorea.go.kr"}:
        return str(value).strip()
    return ""


# ---------------- 재료 기반 추천 (v0.2 로직 + 순위 보정) ----------------
PANTRY = {"물", "소금", "후추", "식용유", "설탕"}
OPTIONAL_SECTIONS = {"SEASONING", "BROTH", "GARNISH"}
ALIASES = {"닭": "닭고기", "계란": "달걀", "파": "대파", "쇠고기": "소고기", "돼지": "돼지고기"}


def tokens(value: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,，、\n]", value or "") if x.strip()]


def canonical(value: str) -> str:
    v = ALIASES.get(value.strip(), value.strip())
    items = parse("입력", v + " 1g")["items"]
    if items:
        return str(items[0].get("base") or items[0].get("norm") or v).replace(" ", "").lower()
    return v.replace(" ", "").lower()


def rank_recipes(recipes, owned, category="전체", search="", max_missing=None, pantry_basics=False, sort_by="추천순"):
    typed = {canonical(t) for t in owned}
    owned_set = set(typed) | (PANTRY if pantry_basics and typed else set())
    out = []
    for r in recipes:
        if category != "전체" and r["category"] != category: continue
        if search and search.casefold() not in r["name"].casefold(): continue
        uniq = {}
        for i in r["ingredients"]:
            k = str(i.get("base") or i.get("norm") or i.get("name") or "").replace(" ", "").lower()
            if k and k not in uniq:
                uniq[k] = {"name": str(i.get("name") or k), "section": str(i.get("section") or "UNSPECIFIED")}
        if not typed:
            out.append({"recipe": r, "matched": [], "missing": [], "missing_essential": [], "missing_optional": [],
                        "score": 0, "used_inputs": 0, "required_count": len(uniq)})
            continue
        if not uniq: continue
        matched = [k for k in uniq if k in owned_set]
        if not any(k in typed for k in matched): continue          # 입력한 재료를 하나도 쓰지 않으면 제외
        missing = [k for k in uniq if k not in owned_set]
        if max_missing is not None and len(missing) > max_missing: continue
        ess = [k for k, v in uniq.items() if v["section"] not in OPTIONAL_SECTIONS] or list(uniq)
        ess_ratio = sum(k in owned_set for k in ess) / len(ess)
        used_inputs = sum(1 for t in typed if t in uniq)
        input_cover = used_inputs / len(typed)
        # v0.3 보정: 입력한 재료를 얼마나 쓰는지(40%)를 반영해 '주재료 1개짜리' 레시피가 위로 뜨는 문제를 줄인다.
        score = round(100 * (0.45 * ess_ratio + 0.15 * len(matched) / len(uniq) + 0.40 * input_cover))
        out.append({"recipe": r, "matched": [uniq[k]["name"] for k in matched],
                    "missing": [uniq[k]["name"] for k in missing],
                    "missing_essential": [uniq[k]["name"] for k in missing if k in ess],
                    "missing_optional": [uniq[k]["name"] for k in missing if k not in ess],
                    "score": score, "used_inputs": used_inputs, "required_count": len(uniq)})
    if not typed:
        return sorted(out, key=lambda x: x["recipe"]["name"])
    if sort_by == "부족 재료 적은 순":
        key = lambda x: (len(x["missing_essential"]), len(x["missing"]), -x["score"], x["recipe"]["name"])
    elif sort_by == "보유 재료 많이 쓰는 순":
        key = lambda x: (-x["used_inputs"], len(x["missing_essential"]), -x["score"], x["recipe"]["name"])
    else:
        key = lambda x: (-x["score"], -x["used_inputs"], len(x["missing_essential"]), x["recipe"]["name"])
    return sorted(out, key=key)
