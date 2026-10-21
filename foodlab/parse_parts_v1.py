"""RCP_PARTS_DTLS → 파생 재료 목록 v1 (원본 불변). v0 대비 변경점은 CHANGES 참고. 승인 전 확정 아님.

출력 항목 필드
  raw      원문 토큰 (추적용, 원본에서 그대로 잘라낸 문자열)
  line     원문 줄 번호(0부터, <br>을 줄바꿈으로 본 기준)
  name     원문 재료명 (괄호·수량 제거만)
  form     형태 표현 (다진/건/불린/채썬 …)
  norm     엄격 동의어 정규명 — 같은 물건의 다른 표기만 합침 (계란=달걀, 쇠고기=소고기, 브로컬리=브로콜리)
  base     검색용 상위 재료 — 형태·가공을 벗긴 재료 계열 (다진 마늘→마늘, 레몬즙→레몬, 저염간장→간장)
  qty/unit 수량·단위 (약간·적당량은 qty 문자열)
  section  MAIN / SUB / SEASONING / BROTH / GARNISH / DOUGH / FILLING / OTHER_SECTION / UNSPECIFIED
"""
import re

RULE_VERSION = "parts-v1.1-20261022"
CHANGES = [
    "V1-1 줄 머리 구역명(+공백) 인식: '육수 무(50g)', '양념장 고춧가루(1g)', '밥 밑간 참기름', '…드레싱 양파', '…소스 설탕'",
    "V1-2 'n인분 기준', '[n인분]' 안내 문구 제거 → flags",
    "V1-3 숫자 없는 단독 줄이 구역어로 끝나면 헤더(주재료·양념·두유 소스 등), 아니면 메뉴명 반복 줄",
    "V1-4 '…장'(초고추장·초간장·곁들이장·재움장)·'재움'·'밑간' 구역은 SEASONING 우선",
    "V1-5 단위 ㎖ ℓ ㎏ ㎎ 추가",
    "V1-6 접미 형태(다진것·마른것·채썬것·얇게썬것·불린것) 분리, 표기 흔들림 사전 확장, norm/base 이중 필드",
    "V1.1-1 '주재료 >', '쑥크림 >' 처럼 '>' 로 구분한 구역 인식 (독립 표본 2·PDF 대조에서 발견)",
    "V1.1-2 '[겉절이 양념(124g분량)]' 처럼 괄호 안 분량이 붙은 대괄호 구역 인식",
    "V1.1-4 재료명 뒤 ①② 구분 기호는 note 로 이동",
    "V1.1-3 수량 뒤에 단독으로 붙은 구역어(… 5g 장식) 를 재료가 아니라 구역 전환으로 처리",
]
UNITS = r"(?:kg|g|mg|ml|mL|L|l|cc|㎖|ℓ|㎏|㎎|개|컵|큰술|작은술|T|t|ts|Ts|스푼|숟가락|장|줄기|쪽|알|마리|모|송이|뿌리|포기|단|줌|꼬집|토막|조각|대|봉|팩|캔|통|잎|cm|인분|봉지|주먹)"
QTY_WORDS = r"(?:약간|적당량|적량|조금|소량|적당히|약간씩|기호에\s*따라)"
SECTION_WORDS = r"(?:양념장|양념|소스|드레싱|육수|고명|장식|곁들임|곁들이|토핑|주재료|부재료|속재료|반죽|밑간|재움장|초간장|초고추장|시럽|튀김옷|재료)"

def _role(label):
    label = re.sub(r"[\s●•·\-\[\]()<>:：]", "", label)
    if not label: return "OTHER_SECTION"
    if re.search(r"(장|소스|드레싱|양념|밑간|재움|절임|시럽|소스소개)$", label) or "절임" in label or "양념" in label or "소스" in label or "드레싱" in label:
        return "SEASONING"
    if "육수" in label or "국물" in label: return "BROTH"
    if re.search(r"고명|장식|곁들임|곁들이|토핑", label): return "GARNISH"
    if "반죽" in label: return "DOUGH"
    if re.search(r"속재료|^소$|속$", label): return "FILLING"
    if "부재료" in label or "완자" in label or "튀김옷" in label: return "SUB"
    if re.search(r"주재료|필수재료|^재료$|간편식재료|추가재료", label): return "MAIN"
    return "OTHER_SECTION"

# 엄격 동의어: 같은 물건의 다른 표기
NORM = {"계란": "달걀", "쇠고기": "소고기", "브로컬리": "브로콜리", "쭈꾸미": "주꾸미", "가쯔오부시": "가쓰오부시",
        "가다랑어포": "가쓰오부시", "토마토페스트": "토마토페이스트", "토마토 페스트": "토마토페이스트",
        "모짜렐라치즈": "모차렐라치즈", "모짜렐라 치즈": "모차렐라치즈", "모짜렐라": "모차렐라치즈",
        "파마산치즈": "파르메산치즈", "파마산 치즈": "파르메산치즈", "파르메산 치즈": "파르메산치즈",
        "후춧가루": "후추", "흰후춧가루": "흰후추", "흰 후춧가루": "흰후추", "올리브오일": "올리브유",
        "녹말가루": "전분", "녹말": "전분", "감자전분": "감자전분", "붉은 고추": "홍고추", "붉은고추": "홍고추",
        "양송이": "양송이버섯", "양송이 버섯": "양송이버섯", "닭 가슴살": "닭가슴살", "계란노른자": "달걀노른자",
        "달걀 노른자": "달걀노른자", "계란 노른자": "달걀노른자", "정종": "청주", "요플레": "요거트", "요구르트": "요거트",
        "플레인 요거트": "플레인요거트", "케찹": "케첩", "쥬키니호박": "주키니호박", "쥬키니": "주키니호박",
        "래디쉬": "래디시", "청고주": "청고추", "와사미": "와사비", "샄": "살"}
# 검색용 상위 재료 계열 (정보 손실 있음 → base 에만 적용)
BASE = {"후추": "후추", "흰후추": "후추", "통후추": "후추", "다진마늘": "마늘", "다진 마늘": "마늘", "편 마늘": "마늘",
        "편마늘": "마늘", "알마늘": "마늘", "마늘기름": "마늘기름", "다진파": "대파", "다진 파": "대파", "파": "대파",
        "쪽파": "쪽파", "참깨": "깨", "통깨": "깨", "검은깨": "검은깨", "깨소금": "깨",
        "저염간장": "간장", "저염 간장": "간장", "양조간장": "간장", "진간장": "간장", "국간장": "간장", "맛간장": "간장",
        "어간장": "간장", "저염된장": "된장", "저염 된장": "된장", "천일염": "소금", "함초소금": "소금", "꽃소금": "소금",
        "레몬즙": "레몬", "레몬주스": "레몬", "레몬껍질": "레몬", "새우살": "새우", "칵테일새우": "새우", "건새우": "새우",
        "냉동 새우살": "새우", "보리새우": "새우", "요리당": "올리고당", "포도씨유": "식용유", "카놀라유": "식용유",
        "콩기름": "식용유", "현미유": "식용유", "닭가슴살": "닭고기", "닭고기살": "닭고기", "닭다리살": "닭고기",
        "소고기 양지": "소고기", "쇠고기등심": "소고기", "돼지고기살코기": "돼지고기", "통삼겹": "삼겹살",
        "감자전분": "전분", "옥수수전분": "전분", "쌀밥": "밥", "현미밥": "밥", "홍파프리카": "파프리카",
        "노랑파프리카": "파프리카", "빨강파프리카": "파프리카", "노란색파프리카": "파프리카", "적파프리카": "파프리카",
        "황파프리카": "파프리카", "청피망": "피망", "홍피망": "피망", "붉은피망": "피망", "적양파": "양파",
        "깐양파": "양파", "양파채썬것": "양파", "새송이버섯": "새송이버섯", "건표고버섯": "표고버섯",
        "건미역": "미역", "불린 미역": "미역", "조선부추": "부추", "영양부추": "부추", "호부추": "부추",
        "생크림": "생크림", "무염버터": "버터", "저염버터": "버터", "저염치즈": "치즈", "슬라이스치즈": "치즈",
        "치즈슬라이스": "치즈", "통밀가루": "밀가루", "박력분": "밀가루", "강력분": "밀가루", "중력분": "밀가루",
        "우리밀가루": "밀가루", "밀가루중력분": "밀가루", "강력 밀가루": "밀가루", "찹쌀가루": "찹쌀가루",
        "방울토마토": "토마토", "홀토마토": "토마토", "밤고구마": "고구마", "자색고구마": "고구마", "통배추": "배추",
        "알배추": "배추", "알배기배추": "배추", "얼갈이배추": "얼갈이배추", "훈제연어": "연어", "훈재연어": "연어"}
FORM_PREFIX = r"^(?:다진\s*|(?:간|채썬|썬|불린|삶은|데친|냉동|건|말린|생|슬라이스|저염|무염|저지방|볶은|구운|으깬|깐|편)\s+)"
FORM_SUFFIX = r"(다진것|마른것|채썬것|얇게썬것|썬것|불린것|삶은것|데친것|간것)$"

def _split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([": depth += 1
        elif ch in ")]": depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return out

HEADER_COLON = re.compile(r"^\s*[●•·\-\*▶■◆○]?\s*[\[\(<]?\s*([가-힣A-Za-z ]{1,15}?)\s*[\]\)>]?\s*[:：]\s*")
BRACKET_HEAD = re.compile(r"^\s*[●•·\-]?\s*[\[\(]\s*([가-힣 ]{1,12})\s*(?:\([^)\]]{0,15}\))?\s*[\]\)]\s*")
GT_HEAD = re.compile(r"^\s*([가-힣][가-힣 ]{0,11}?)\s*>\s*")                                    # V1.1-1
SECTION_ONLY = re.compile(r"^" + "(?:양념장|양념|소스|드레싱|육수|고명|장식|곁들임|토핑)" + r"$")
SERVING = re.compile(r"\[?\s*\d+\s*인분\s*(?:기준)?\s*\]?")
# V1-1: 줄 머리 '구역명 + 공백 + 재료명(' 패턴.  구역명은 SECTION_WORDS 로 끝나는 1~14자
LINE_HEAD = re.compile(r"^((?:[가-힣]{1,10}\s){0,2}[가-힣]{0,10}?" + SECTION_WORDS + r")\s+(?=[가-힣])")

def parse(rcp_nm, text):
    items, unparsed, flags = [], [], []
    if not (text or "").strip():
        return {"rule_version": RULE_VERSION, "items": [], "unparsed": [], "flags": ["EMPTY"]}
    t = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    t = re.sub(r"\s*(\[[가-힣 ]{1,12}(?:\([^)\]]{0,15}\))?\])\s*", r"\n\1 ", t)
    t = re.sub(r"(?<=[,\d)g])\s*([가-힣][가-힣 ]{0,11}?)\s*>\s*", r"\n\1 > ", t)          # V1.1-1 (줄 중간 '구역 >')
    t = re.sub(r"\s*(\((?:반죽|속|소스|양념|고명|토핑)[가-힣 ]{0,6}\))\s*", r"\n\1 ", t)
    t = re.sub(r"(?<=[\dg)a-z가-힣])\s+(?=(?:양념장?|소스|드레싱|육수|고명|장식|곁들임|토핑|주재료|부재료)\s*[:：])", "\n", t)
    name_c = re.sub(r"\s", "", rcp_nm)
    section = "UNSPECIFIED"
    for li, line in enumerate(t.split("\n")):
        line = line.strip().strip(",")
        if not line: continue
        for m in SERVING.finditer(line): flags.append("SERVING:" + re.sub(r"\D", "", m.group(0)) + "인분")
        line = SERVING.sub("", line).strip(" •·-")
        if not line: continue
        compact = re.sub(r"\s", "", line)
        if not re.search(r"\d|약간|적당", line) and len(compact) <= 20 and "," not in line:
            label = re.sub(r"[●•·\-\[\]()<>:：]", "", compact)
            if re.search(SECTION_WORDS + r"$", label) or re.search(r"재료$", label):   # V1-3
                section = _role(label); flags.append("HEADER:" + label); continue
            if label and (label in name_c or name_c in label or li == 0):
                flags.append("TITLE_LINE"); continue
            section = _role(label); flags.append("HEADER?:" + label); continue
        m = HEADER_COLON.match(line)
        if m and not re.search(r"\d", m.group(1)):
            section = _role(m.group(1)); line = line[m.end():]
        mb = BRACKET_HEAD.match(line)
        if mb and not re.search(r"\d", mb.group(1)):
            section = _role(mb.group(1)); line = line[mb.end():]
        mg = GT_HEAD.match(line)
        if mg and not re.search(r"\d", mg.group(1)):
            section = _role(mg.group(1)); flags.append("HEADER:" + mg.group(1).strip()); line = line[mg.end():]
        mh = LINE_HEAD.match(line)                                                   # V1-1
        if mh:
            section = _role(mh.group(1)); flags.append("HEADER:" + mh.group(1).strip()); line = line[mh.end():]
        line = re.sub(r"(\d(?:\.\d+)?\s*" + UNITS + r"?\)?)\s+(?=[가-힣])", r"\1,", line)
        for tok in _split_top(line):
            tok = tok.strip(" .·-•")
            if not tok: continue
            mc = re.match(r"^([가-힣 ]{1,14}?)\s*[:：]\s*(.+)$", tok)
            if mc and not re.search(r"\d", mc.group(1)):
                section = _role(mc.group(1)); tok = mc.group(2)
            if SECTION_ONLY.match(tok):                      # V1.1-3
                section = _role(tok); flags.append("HEADER:" + tok); continue
            mh2 = LINE_HEAD.match(tok)                       # 쉼표 뒤에서 새 구역이 시작되는 경우
            if mh2 and _item(tok[mh2.end():]):
                section = _role(mh2.group(1)); tok = tok[mh2.end():]
            it = _item(tok)
            if it:
                it["section"] = section; it["line"] = li; items.append(it)
            else:
                unparsed.append(tok)
    return {"rule_version": RULE_VERSION, "items": items, "unparsed": unparsed, "flags": flags}

def _item(tok):
    raw, note, qty_part = tok, None, None
    m = re.match(r"^([^\d(]+?)\s*\((.*)\)\s*$", tok)
    if m:
        name, inner = m.group(1), m.group(2)
        parts = [p.strip() for p in inner.split(",")]
        q = [p for p in parts if re.search(r"\d|" + QTY_WORDS, p)]
        n = [p for p in parts if p not in q]
        qty_part = q[-1] if q else None
        note = ", ".join(n) or None
    else:
        m = re.match(r"^(.+?)\s*((?:\d|½|⅓|¼|" + QTY_WORDS + r").*)$", tok)
        if m: name, qty_part = m.group(1), m.group(2)
        else: name = tok
    name = re.sub(r"\(.*?\)", "", name).strip(" :")
    mk = re.search(r"[①-⑳]", name)
    if mk: name = name.replace(mk.group(0), "").strip(); note = (note + ", " if note else "") + "구분" + mk.group(0)
    name = re.sub(QTY_WORDS + "$", "", name).strip()
    if not name or re.fullmatch(r"[\d\W]+", name) or len(name) > 20: return None
    qty = unit = None
    if qty_part:
        mq = re.search(r"([\d./~½⅓¼⅔¾]+(?:\s*[\d./½⅓¼]+)?)\s*(" + UNITS + r")?", qty_part)
        if mq: qty, unit = mq.group(1), mq.group(2)
        elif re.search(QTY_WORDS, qty_part): qty = re.search(QTY_WORDS, qty_part).group(0)
    stem, form = name, None
    mf = re.match(FORM_PREFIX, stem)
    if mf and len(stem) > len(mf.group(0)): form = mf.group(0).strip(); stem = stem[mf.end():]
    ms = re.search(FORM_SUFFIX, stem)
    if ms and len(stem) > len(ms.group(0)): form = (form + "," if form else "") + ms.group(0); stem = stem[:ms.start()]
    norm = NORM.get(name, NORM.get(stem, stem)).replace("샄", "살")
    base = BASE.get(name, BASE.get(norm, BASE.get(stem, norm)))
    return {"raw": raw, "name": name, "form": form, "norm": norm, "base": base, "note": note, "qty": qty, "unit": unit}
