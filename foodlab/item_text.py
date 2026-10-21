"""F2001: 레시피 1건 → 검색용 텍스트 1개 + 메타데이터 (항목당 1청크)."""
from __future__ import annotations

import hashlib

from .data import NUTRITION_FLAGS

TEXT_VERSION = "item-text-v1"
SEASON_SECTIONS = {"SEASONING", "BROTH"}


def _fmt(v, unit):
    if v is None: return "정보 없음"
    return f"{v:g}{unit}"


def build_text(r: dict) -> str:
    main = [i["name"] for i in r["ingredients"] if i.get("section") not in SEASON_SECTIONS and i.get("section") != "GARNISH"]
    seas = [i["name"] for i in r["ingredients"] if i.get("section") in SEASON_SECTIONS]
    garn = [i["name"] for i in r["ingredients"] if i.get("section") == "GARNISH"]
    n = r["nutrition"]
    lines = [
        f"[레시피] {r['name']} (레시피 번호 {r['id']})",
        f"요리 종류: {r['category']} / 조리 방법: {r['method']}",
        f"주요 재료: {', '.join(dict.fromkeys(main)) or '정보 없음'}",
    ]
    if seas: lines.append(f"양념·육수 재료: {', '.join(dict.fromkeys(seas))}")
    if garn: lines.append(f"고명·곁들임: {', '.join(dict.fromkeys(garn))}")
    lines.append("재료 원문: " + (r["ingredient_text"].replace("\n", " / ") if r["ingredient_text"] else "원본에 재료 정보 없음"))
    if r["steps"]:
        lines.append("만드는 법: " + " ".join(f"{k}) {s.replace(chr(10), ' ')}" for k, s in enumerate(r["steps"], 1)))
    else:
        lines.append("만드는 법: 원본에 조리 단계 정보 없음")
    nut = (f"영양정보(원본 표기, 1인분): 열량 {_fmt(n['kcal'], 'kcal')}, 탄수화물 {_fmt(n['carb_g'], 'g')}, "
           f"단백질 {_fmt(n['protein_g'], 'g')}, 지방 {_fmt(n['fat_g'], 'g')}, 나트륨 {_fmt(n['sodium_mg'], 'mg')}")
    if set(r["flags"]) & NUTRITION_FLAGS:
        nut += " [주의: 원본 영양 수치에 이상치가 있어 검토가 필요함]"
    lines.append(nut)
    if r["tip"]: lines.append("저감 조리 팁: " + r["tip"].replace("\n", " "))
    if r["hashtag"]: lines.append("해시태그: " + r["hashtag"])
    lines.append("출처: 식품의약품안전처 식품안전나라 조리식품의 레시피 DB(COOKRCP01)")
    return "\n".join(lines)


def build_metadata(r: dict, raw_hashes: dict) -> dict:
    n = r["nutrition"]
    md = {
        "rcp_seq": r["id"], "name": r["name"], "display_name": r["display_name"],
        "category": r["category"], "method": r["method"],
        "source_type": "recipe_meta", "source_label": "레시피 메타데이터(식품안전나라)",
        "source_url": r["source"], "flags": ",".join(r["flags"]),
        "nutrition_suspect": bool(set(r["flags"]) & NUTRITION_FLAGS),
        "main_ingredients": ",".join(dict.fromkeys(i.get("base") or i["name"] for i in r["ingredients"]
                                                   if i.get("section") not in SEASON_SECTIONS))[:1000],
        "text_version": TEXT_VERSION, "raw_sha256": ",".join(f"{k}:{v[:12]}" for k, v in sorted(raw_hashes.items())),
    }
    for k in ("kcal", "carb_g", "protein_g", "fat_g", "sodium_mg"):
        if n[k] is not None: md[k] = n[k]
    return md


def content_hash(text: str, md: dict, model: str) -> str:
    stable = {k: v for k, v in md.items() if k != "content_hash"}
    return hashlib.sha256((model + "\n" + text + "\n" + repr(sorted(stable.items()))).encode()).hexdigest()[:16]
