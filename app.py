"""Food Lab v0.3 — 내 냉장고 속 AI 셰프 (RAG)

    streamlit run app.py

화면 구성
- 💬 AI 셰프에게 묻기 (메인, 첫 화면)
    · 레시피 선택 없음 → 일반 질문: 레시피 메타데이터 Top-5 + 수집된 위키 문서 → LangGraph retrieve/generate, 근거 레시피 카드로 이어서 대화
    · 레시피 선택 (F2009) → F2003~F2005 수집 → LangGraph retrieve/generate → 인용 근거만 표시(F2008)
- 🥬 재료로 찾기 (v0.2 재료 추천 유지)        → 카드에서 'AI 셰프에게 묻기'로 대화 연결
- 🔎 조건으로 찾기 (F2011 자연어 조건 탐색)    → LLM 조건 해석 + 메타데이터 필터 + 의미 검색
"""
from __future__ import annotations

import html
import re
import time

import streamlit as st

from foodlab.data import NUTRITION_FLAGS, load_recipes, rank_recipes, safe_image_url, tokens
from foodlab.settings import find_raw_dir, gms_key, load_config, wiki_contact

st.set_page_config(page_title="Food Lab | 내 냉장고 속 AI 셰프", page_icon="🥗", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:3.2rem;max-width:1280px}
.hero{background:linear-gradient(115deg,#123e35,#267d5e);color:white;padding:26px 32px;border-radius:20px;margin-bottom:14px}
.hero h1{font-size:2.0rem;color:white;margin:6px 0}.hero p{color:#e4f4e9;margin:0}
.eyebrow{font-weight:800;letter-spacing:.13em;font-size:.76rem;color:#c0e5d5}
.recipe-photo{width:100%;height:170px;object-fit:cover;border-radius:12px;background:#e7eee9}
.recipe-placeholder{height:170px;border-radius:12px;background:linear-gradient(145deg,#e7f3eb,#d3e7dc);display:flex;align-items:center;justify-content:center;color:#41715b;font-size:42px}
.recipe-title{font-size:1.08rem;font-weight:750;margin:8px 0 4px}
.st-key-hero{background:linear-gradient(115deg,#123e35,#267d5e);padding:26px 32px 24px;border-radius:20px;margin-bottom:6px;gap:14px !important}
.st-key-hero [data-testid="stMarkdownContainer"] p,.st-key-hero .eyebrow{margin:0 !important}
.st-key-home button{background:transparent !important;border:none !important;padding:0 !important;min-height:0 !important;
 justify-content:flex-start;box-shadow:none !important}
.st-key-home button p{font-size:2rem !important;font-weight:800 !important;color:#fff !important;line-height:1.3}
.st-key-home button:hover p{text-decoration:underline;text-underline-offset:6px}
.hero-sub{color:#e4f4e9;margin:0 !important}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:4px 0 14px}
.step{background:#f3f8f5;border:1px solid #d9e9df;border-radius:14px;padding:10px 14px}
.step b{display:block;color:#16523f;font-size:.95rem}.step span{color:#4b5f56;font-size:.84rem}
.step .n{display:inline-block;background:#267d5e;color:#fff;border-radius:50%;width:20px;height:20px;text-align:center;
 font-size:.78rem;line-height:20px;margin-right:6px}
.pagehint{color:#4b5f56;font-size:.92rem;margin:-4px 0 12px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.pagehint .idx{color:#2f6f57;background:#eef6f1;border-radius:999px;padding:2px 12px;font-size:.84rem}
@media (max-width:700px){.steps{grid-template-columns:1fr}}
div[role="radiogroup"]{gap:8px;flex-wrap:wrap}
div[role="radiogroup"] > label{border:1px solid #cfe1d6;border-radius:999px;padding:7px 16px;background:#fff;margin:0}
div[role="radiogroup"] > label:has(input:checked){background:#267d5e;border-color:#267d5e}
div[role="radiogroup"] > label:has(input:checked) p{color:#fff;font-weight:700}
div[role="radiogroup"] > label > div:first-child{display:none}
div[role="radiogroup"] p{font-size:1rem}
/* 입력칸: 테두리·크기를 키워 눈에 띄게 */
[data-testid="stChatInput"]{border:2px solid #267d5e !important;border-radius:16px !important;background:#fff !important;
 box-shadow:0 6px 18px rgba(38,125,94,.18)}
[data-testid="stChatInput"] > div{background:#fff !important;border:none !important}
[data-testid="stChatInput"] textarea{font-size:1.08rem !important;min-height:60px !important;padding-top:18px !important;color:#16241e}
[data-testid="stChatInput"] textarea::placeholder{color:#5d7a6c}
[data-testid="stTextInputRootElement"],[data-testid="stSelectbox"] div[role="group"]{
 border:2px solid #6fb294 !important;border-radius:12px !important;background:#fff !important;min-height:50px}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stSelectbox"] div[role="group"]:focus-within{
 border-color:#267d5e !important;box-shadow:0 0 0 3px rgba(38,125,94,.18)}
[data-testid="stTextInputField"],[data-testid="stSelectbox"] input{font-size:1.05rem !important;background:#fff !important;color:#16241e}
[data-testid="stTextInputField"]{padding:12px 14px !important}
[data-testid="stTextInputField"]::placeholder,[data-testid="stSelectbox"] input::placeholder{color:#5d7a6c}
[data-testid="stTextInput"] label p,[data-testid="stSelectbox"] label p{font-weight:700;color:#16523f;font-size:1rem}
[data-testid="stBaseButton-primary"]{background:#267d5e !important;border-color:#267d5e !important;font-weight:700;padding:8px 22px}
[data-testid="stBaseButton-primary"]:hover{background:#1d6249 !important}
.ask-label{font-weight:750;color:#16523f;margin:10px 0 4px;font-size:1.02rem}
</style>""", unsafe_allow_html=True)

HEALTH = "건강·영양 정보는 의료적 조언을 대신하지 않습니다. 알레르기·질환이 있다면 전문가와 상담하세요. 영양 수치는 원본 표기이며 일부 오류가 있을 수 있습니다."
FLAG_KO = {"STEPS_EMPTY": "조리 단계 없음", "STEPS_NUMBER_GAP": "조리 단계 번호 누락", "PARTS_EMPTY": "재료 정보 없음",
           "NUTRITION_ALL_ZERO": "영양 정보 0", "KCAL_MACRO_MISMATCH_HIGH": "열량·탄단지 불일치", "KCAL_MACRO_MISMATCH_LOW": "열량·탄단지 불일치",
           "MACRO_IMPLAUSIBLE_GT300G": "비현실적 영양 수치", "SODIUM_IMPLAUSIBLE": "비현실적 나트륨 수치"}
REL_KO = {"FOOD_BACKGROUND": "음식 배경", "INGREDIENT_BACKGROUND": "재료 배경", "COOKING_METHOD": "조리법"}


# ---------------- 데이터·자원 (캐시) ----------------
@st.cache_data(show_spinner=False)
def _recipes(raw_dir: str):
    from pathlib import Path
    return load_recipes(Path(raw_dir))


@st.cache_resource(show_spinner=False)
def _resources(has_key: bool, config_text: str):
    """키·색인 상태를 점검하고 사용할 수 있는 자원만 돌려준다 (F2012)."""
    cfg = load_config()
    out = {"cfg": cfg, "problems": [], "recipe_col": None, "ext_col": None, "emb": None, "chat": None, "graph": None, "general": None}
    if not gms_key():
        out["problems"].append(("key", "GMS_KEY 가 설정되지 않았습니다. `.env` 에 `GMS_KEY=...` 를 넣고 앱을 다시 실행하세요. (AI 기능 비활성)"))
        return out
    if not cfg.get("embedding_model"):
        out["problems"].append(("model", "임베딩 모델이 설정되지 않았습니다. `python check_gms.py --save` 를 먼저 실행하세요."))
        return out
    from foodlab.embedder import GMSEmbedder
    from foodlab.llm import get_chat
    from foodlab.rag import build_general_graph, build_graph
    from foodlab.store import StoreError, get_collection
    try:
        out["emb"] = GMSEmbedder(cfg, max_requests=10_000)
        out["chat"] = get_chat(cfg)
        rc = get_collection(cfg, cfg["recipe_collection"], out["emb"].model)
        if rc is None or rc.count() == 0:
            out["problems"].append(("index", "레시피 색인이 없습니다. 터미널에서 `python build_index.py` 를 실행한 뒤 아래 '상태 다시 확인'을 누르세요."))
            return out
        out["recipe_col"] = rc
        out["ext_col"] = get_collection(cfg, cfg["external_collection"], out["emb"].model, create=True)
        out["graph"] = build_graph(cfg, rc, out["ext_col"], out["emb"], out["chat"])
        out["general"] = build_general_graph(cfg, rc, out["ext_col"], out["emb"], out["chat"])
    except StoreError as e:
        out["problems"].append(("index", str(e)))
    except Exception as e:
        out["problems"].append(("init", f"AI 기능 초기화 실패 ({type(e).__name__}). 설정을 확인하세요."))
    return out


raw_dir = find_raw_dir()
if not raw_dir:
    st.error("원본 레시피(data_check/raw/cookrcp01_*.json)를 찾지 못했습니다. food_lab_v0_3 폴더를 streamlit_lab 안에 두세요.")
    st.stop()
recipes, raw_hashes = _recipes(str(raw_dir))
R = {r["id"]: r for r in recipes}
from foodlab.settings import PROJECT as _P
import os as _os
_cfg_path = _os.getenv("FOODLAB_CONFIG") or str(_P / "config.json")
from pathlib import Path as _Path
res = _resources(bool(gms_key()), _Path(_cfg_path).read_text(encoding="utf-8"))
ai_ready = res["graph"] is not None

ss = st.session_state
P_CHAT, P_ING, P_NL = "💬 AI 셰프에게 묻기", "🥬 내 재료로 찾기", "🔎 조건으로 찾기"
ss.setdefault("page", P_CHAT)
ss.setdefault("fridge_text", "")
ss.setdefault("chats", {})


def tidy_md(t: str) -> str:
    """화면 표시용 정리 (저장된 답변 원문은 그대로): 제목(#)은 별도 문단의 일반 글씨로, 굵은 글씨 기호(**)는 제거.
    한글 조사가 바로 붙은 **…**이 는 마크다운이 강조로 인식하지 못해 별표가 그대로 보이기 때문."""
    t = re.sub(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", r"\n\1\n", t, flags=re.M)
    t = re.sub(r"(?<!\*)\*{2,3}(?!\*)", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def go_home():
    """제목을 누르면 첫 화면(AI 셰프 일반 질문)으로. 대화 기록과 냉장고 재료는 유지한다."""
    ss.page = P_CHAT
    ss.selected_seq = None


def go_general():
    ss.selected_seq = None
    ss.page = P_CHAT


def go_chat(seq: str):
    ss.selected_seq = seq
    ss.page = P_CHAT


def todays_picks(n: int = 3) -> list[dict]:
    """오늘의 추천 메뉴: 날짜로 고정된 무작위(하루 동안 같은 결과) + '다른 추천' 누를 때마다 바뀜.
    사진이 있고 품질 플래그가 없는 레시피만, 요리 종류가 겹치지 않게 고른다."""
    import datetime, random
    rng = random.Random(f"{datetime.date.today().isoformat()}-{ss.get('pick_round', 0)}")
    pool = {}
    for r in recipes:
        if not r["flags"] and safe_image_url(r["image"]):
            pool.setdefault(r["category"], []).append(r)
    cats = rng.sample([c for c in ("반찬", "국&찌개", "밥", "일품", "후식") if pool.get(c)], n)
    return [rng.choice(sorted(pool[c], key=lambda x: x["id"])) for c in cats]


def ask_example(q: str):
    ss.pending_q = q


def fridge_input(where: str, label="🧊 내 냉장고 재료", help_text=None):
    """냉장고 재료 입력칸. 화면마다 따로 그리지만 값은 ss.fridge_text 하나로 공유한다."""
    key = f"fridge_{where}"
    ss[key] = ss.fridge_text
    st.text_input(label, key=key, placeholder="예: 두부, 달걀, 대파 (쉼표로 구분)", help=help_text,
                  on_change=lambda: ss.update(fridge_text=ss[key]))
    return tokens(ss.fridge_text)


def show_steps(m):
    """F2012: 답변이 끝난 뒤에도 처리 과정(수집 단계·수집 문서·소요 시간)을 펼쳐 볼 수 있게 남긴다."""
    if m.get("steps"):
        with st.expander(f"⏱ 처리 과정 · {m.get('summary', '')}"):
            for x in m["steps"]: st.caption("· " + x)


def show_sources(m):
    cited = m.get("cited", [])
    if cited:
        with st.expander(f"📎 답변에 사용한 근거 {len(cited)}건 (출처 확인)"):
            for s_ in cited:
                rel = f" · {REL_KO.get(s_.get('relation'), '')}" if s_.get("relation") else ""
                st.markdown(f"**[근거 {s_['n']}]** {s_['label']}{rel} — {html.escape(str(s_['title']))} · [원문]({s_['url']})")
    else:
        st.caption("ⓘ 답변에 인용된 근거가 없습니다.")
    for n in m.get("notices", []): st.caption("ⓘ " + n)


def recipe_card(r, item=None, reason=None, key_prefix="c", button=True, note=None):
    with st.container(border=True):
        url = safe_image_url(r["image"])
        if url:
            st.markdown(f'<img class="recipe-photo" src="{html.escape(url, quote=True)}" loading="lazy" '
                        f'onerror="this.style.display=\'none\'" alt="{html.escape(r["name"], quote=True)}">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recipe-placeholder">🥗</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recipe-title">{html.escape(r["display_name"])}</div>', unsafe_allow_html=True)
        st.caption(f'{r["category"]} · {r["method"]} · 레시피 #{r["id"]}')
        if reason: st.caption("조건 확인: " + reason)
        if note: st.caption(note)
        if item:
            st.progress(item["score"] / 100, text=f"재료 기반 추천 점수 {item['score']}점 (입력 재료 {item['used_inputs']}개 사용)")
            if item["matched"]: st.markdown("**있는 재료:** " + ", ".join(item["matched"][:10]))
            if item["missing_essential"]: st.markdown("**부족한 주요 재료:** " + ", ".join(item["missing_essential"][:10]))
            if item["missing_optional"]: st.caption("추가 양념·육수·고명: " + ", ".join(item["missing_optional"][:8]))
        if set(r["flags"]) & NUTRITION_FLAGS:
            st.caption("⚠️ 원본 영양 수치 검토 필요")
        if button:
            st.button("💬 이 레시피로 묻기", key=f"{key_prefix}-{r['id']}", on_click=go_chat, args=(r["id"],), width="stretch")


# ---------------- 머리말 · 화면 선택 ----------------
with st.container(key="hero"):
    st.markdown('<div class="eyebrow">FOOD LAB · 내 냉장고 속 AI 셰프</div>', unsafe_allow_html=True)
    st.button("오늘 뭐 해 먹지? 물어보세요", key="home", on_click=go_home, help="첫 화면으로 돌아가기")
    st.markdown('<p class="hero-sub">식품안전나라 공식 레시피 1,156건과 위키백과 문서를 근거로, 출처와 함께 답합니다.</p>',
                unsafe_allow_html=True)
for kind, msg in res["problems"]:
    st.warning(msg, icon="⚠️")
if res["problems"]:
    # 문제가 있을 때만: 색인을 만들었거나 .env 를 고친 뒤 앱을 끄지 않고 다시 점검 (F2012)
    if st.button("🔄 상태 다시 확인", help="색인을 새로 만들었거나 .env 를 고친 뒤 누르세요"):
        _resources.clear(); st.rerun()

page = st.radio("화면", [P_CHAT, P_ING, P_NL], key="page", horizontal=True, label_visibility="collapsed")
HINTS = {P_CHAT: "재료·요리·조리법 무엇이든 질문하세요. 레시피를 고르지 않아도 됩니다.",
         P_ING: "가진 재료를 넣으면 그 재료를 많이 쓰는 공식 레시피를 점수순으로 보여 줍니다.",
         P_NL: "'나트륨 낮은 국'처럼 원하는 조건을 문장으로 적으면 맞는 레시피를 찾아 줍니다."}
_rc_n = res["recipe_col"].count() if res["recipe_col"] else 0
_ec_n = res["ext_col"].count() if res["ext_col"] else 0
st.markdown(f'<div class="pagehint">{HINTS[page]}'
            f'<span class="idx">📚 색인: 레시피 메타데이터 {_rc_n:,}건 · 위키백과 청크 {_ec_n:,}건</span></div>', unsafe_allow_html=True)


# ---------------- 1. AI 셰프에게 묻기 (메인) ----------------
if page == P_CHAT and not ss.get("selected_seq"):
    st.markdown('<div class="steps">'
                '<div class="step"><b><span class="n">1</span>질문하기</b><span>"두부랑 달걀로 뭐 만들지?" 처럼 편하게. 재료는 질문에 써도 되고 오른쪽 칸에 넣어 둬도 됩니다.</span></div>'
                '<div class="step"><b><span class="n">2</span>근거 확인</b><span>답변 아래 📎 근거를 펼치면 레시피 원문·위키백과 출처가 보입니다.</span></div>'
                '<div class="step"><b><span class="n">3</span>레시피로 이어서</b><span>추천 카드의 💬 이 레시피로 묻기를 누르면 그 레시피만 깊게 물어볼 수 있습니다.</span></div>'
                '</div>', unsafe_allow_html=True)
    chat_col, side_col = st.columns([2.2, 1], gap="large")
    with side_col:
        with st.container(border=True):
            owned_g = fridge_input("general")
            if owned_g:
                st.success("반영됨: " + ", ".join(owned_g) + " — 아래 추천 메뉴와 AI 답변이 이 재료 기준으로 바뀝니다.", icon="✅")
            else:
                st.caption("선택 사항 · 재료를 넣고 Enter 를 누르면 추천 메뉴와 AI 답변이 내 재료 기준으로 바뀝니다.")
        with st.container(border=True):
            st.markdown("**🍳 특정 레시피를 바로 묻기**")
            ids = sorted(R, key=lambda s: R[s]["name"])
            pick = st.selectbox("레시피 이름으로 검색", ids, index=None, key="pick_seq", placeholder="예: 된장국",
                                format_func=lambda s: R[s]["display_name"], label_visibility="collapsed")
            st.button("이 레시피와 대화하기", disabled=not pick, width="stretch",
                      on_click=lambda: go_chat(ss.pick_seq) if ss.get("pick_seq") else None)
        if st.button("대화 지우기", key="clear-general", width="stretch"):
            ss.chats["__general__"] = []; st.rerun()

    with chat_col:
        gmsgs = ss.chats.setdefault("__general__", [])

        def render_general(i, m):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant": show_steps(m)
                st.markdown(tidy_md(m["content"]) if m["role"] == "assistant" else m["content"])
                if m["role"] != "assistant": return
                show_sources(m)
                recs = [x for x in m.get("recipes", []) if x in R]
                if recs:
                    st.caption("답변에 나온 레시피 — 💬 를 누르면 그 레시피로 이어서 대화합니다")
                    for col, sq in zip(st.columns(3, gap="small"), recs[:3]):
                        with col: recipe_card(R[sq], key_prefix=f"g{i}")

        if not gmsgs:
            st.markdown("**이렇게 물어보세요** — 눌러서 바로 질문")
            examples = (["내 냉장고 재료로 만들 수 있는 요리 추천해줘"] if owned_g else []) + [
                "두부랑 달걀로 만들 수 있는 반찬 알려줘", "나트륨 낮은 국 추천해줘", "아이 간식으로 좋은 후식 있어?", "된장찌개 짜지 않게 끓이는 법"]
            ex_cols = st.columns(2)
            for k, ex in enumerate(examples[:4]):
                ex_cols[k % 2].button(ex, key=f"ex-{k}", on_click=ask_example, args=(ex,), width="stretch", disabled=not ai_ready)
        for i, m in enumerate(gmsgs): render_general(i, m)
        st.markdown('<div class="ask-label">✍️ 직접 질문하기</div>', unsafe_allow_html=True)
        gq = st.chat_input("AI 셰프에게 질문하기 (예: 두부랑 달걀로 만들 반찬 알려줘)", disabled=not ai_ready)
        gq = gq or ss.pop("pending_q", None)
        if gq:
            gmsgs.append({"role": "user", "content": gq}); render_general(len(gmsgs) - 1, gmsgs[-1])
            from foodlab.rag import history_text
            with st.chat_message("assistant"):
                t0 = time.time()
                with st.status("공식 레시피와 문서에서 근거를 찾는 중...", expanded=True) as s:
                    s.write("레시피 메타데이터 색인과 수집된 위키백과 문서에서 근거 검색 → 답변 생성")
                    try:
                        out = res["general"].invoke({"question": gq, "fridge": ", ".join(owned_g), "history": history_text(gmsgs[:-1])})
                    except Exception as e:      # NF2004: 오류 추적 정보는 화면에 내보내지 않는다
                        print("[general] 답변 실패:", type(e).__name__)
                        out = {"answer": "답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.", "sources": [], "cited": [], "notes": ["FAILED"]}
                    summary = f"근거 {len(out.get('sources', []))}건 검색 · 인용 {len(out.get('cited', []))}건 · {time.time() - t0:.1f}초"
                    s.update(label=summary, state="complete", expanded=False)
            g_steps = ["레시피 메타데이터 Top-5 + 수집된 위키백과 청크 검색",
                       "근거 목록: " + ", ".join(f"[근거 {x['n']}] {x['label']} {x['title']}" for x in out.get("sources", [])) if out.get("sources") else "검색된 근거 없음",
                       "답변 생성 (프롬프트 3)"]
            cited_recs = [s_["rcp_seq"] for s_ in out.get("cited", []) if s_["type"] == "recipe_meta"]
            other = [s_["rcp_seq"] for s_ in out.get("sources", []) if s_["type"] == "recipe_meta" and s_["rcp_seq"] not in cited_recs]
            notices = []
            if "NO_EXTERNAL_DOCS" in out.get("notes", []): notices.append("아직 수집된 외부 문서가 없어 레시피 메타데이터로만 답했습니다.")
            if any(set(R.get(x, {}).get("flags", [])) & NUTRITION_FLAGS for x in cited_recs):
                notices.append("인용된 레시피 중 원본 영양 수치에 이상치가 있는 레시피가 있습니다.")
            gmsgs.append({"role": "assistant", "content": out["answer"], "cited": out.get("cited", []), "notices": notices,
                          "recipes": cited_recs or other, "steps": g_steps, "summary": summary})
            st.rerun()
        if not gmsgs and owned_g:
            # 재료를 넣으면 바로 보이는 변화: 재료 기반 점수 상위 3개 (LLM 호출 없음, v0.2 추천 로직)
            h1, h2 = st.columns([3, 1])
            h1.markdown(f"**🧊 내 재료로 만들 수 있는 메뉴** — {', '.join(owned_g)} 를 많이 쓰는 순")
            h2.button("더 보기 →", width="stretch", on_click=lambda: ss.update(page=P_ING))
            top = rank_recipes(recipes, owned_g, "전체", "", None, False, "추천순")[:3]
            if not top:
                st.info("입력한 재료를 쓰는 공식 레시피를 찾지 못했습니다. 재료 이름을 바꿔 보세요 (예: 계란 → 달걀).")
            for col, it in zip(st.columns(3, gap="small"), top):
                with col: recipe_card(it["recipe"], it, key_prefix="mine")
        elif not gmsgs:
            h1, h2 = st.columns([3, 1])
            h1.markdown("**🍽️ 오늘의 추천 메뉴** — 공식 레시피에서 무작위로 골랐어요")
            h2.button("🔄 다른 추천", width="stretch", on_click=lambda: ss.update(pick_round=ss.get("pick_round", 0) + 1))
            for col, rr in zip(st.columns(3, gap="small"), todays_picks()):
                n_ = rr["nutrition"]
                note = f"열량 {n_['kcal']:g}kcal · 나트륨 {n_.get('sodium_mg') or 0:g}mg (원본 값)" if n_.get("kcal") is not None else None
                with col: recipe_card(rr, key_prefix="today", note=note)

# ---------------- 1-b. 레시피 하나와 대화 (F2009 + F2003~F2008 + F2010) ----------------
elif page == P_CHAT:
    seq = ss.selected_seq
    r = R[seq]
    top1, top2 = st.columns([1, 4])
    top1.button("← 전체 질문으로", on_click=go_general, width="stretch")
    top2.markdown(f"### 💬 {html.escape(r['display_name'])} — 이 레시피에 대해 묻기")
    left, right = st.columns([1, 2], gap="large")
    with left:
        recipe_card(r, key_prefix="chat", button=False)
        with st.expander("레시피 원문 (재료·조리법)"):
            st.text(r["ingredient_text"] or "공식 원본에 재료 정보가 없습니다.")
            for k, s_ in enumerate(r["steps"], 1): st.write(f"{k}. {s_}")
            if r["tip"]: st.info("공식 조리 팁: " + r["tip"])
            st.link_button("식품안전나라 원문 보기 ↗", r["source"])
        if r["flags"]:
            st.warning("데이터 품질 검토 대상: " + ", ".join(dict.fromkeys(FLAG_KO.get(f, f) for f in r["flags"])))
        from foodlab.collect import STATUS_LABEL, item_status
        stt = item_status(seq)
        if stt.get("item_self"):
            st.caption("외부 문서 상태: " + STATUS_LABEL.get(stt["item_self"], stt["item_self"]))
        if stt.get("docs"):
            st.caption("수집된 문서: " + " · ".join(stt["docs"].keys()))
    with right:
        msgs = ss.chats.setdefault(seq, [])
        with st.container(border=True):
            owned_r = fridge_input("recipe", label="🧊 내 냉장고 재료 (선택)")
            fA, fB = st.columns([3, 1])
            fridge_btn = fA.button("이 재료로 만들 수 있을까? — 있는 것·부족한 것 정리", disabled=not ai_ready or not owned_r,
                                   help=None if owned_r else "위 칸에 재료를 넣으면 눌러 볼 수 있습니다", width="stretch")
            if fB.button("대화 지우기", width="stretch"):
                ss.chats[seq] = []; st.rerun()
        if not msgs:
            st.caption("예: 이 요리 짜지 않게 하려면? / ○○ 대신 쓸 수 있는 재료는? / 이 요리는 어떤 음식이야?")

        def render(m):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant": show_steps(m)
                st.markdown(tidy_md(m["content"]) if m["role"] == "assistant" else m["content"])
                if m["role"] == "assistant": show_sources(m)

        for m in msgs: render(m)
        q = st.chat_input(f"{r['name']}에 대해 물어보세요", disabled=not ai_ready)
        shown_q = q
        if fridge_btn:
            from foodlab.features import fridge_question
            q = fridge_question(owned_r)                                   # LLM 에 보내는 전체 지시문
            shown_q = f"🧊 내 냉장고 재료({', '.join(owned_r)})로 이 요리를 만들 수 있을까?"   # 화면에는 짧게 (NF2004)
        if q:
            msgs.append({"role": "user", "content": shown_q}); render(msgs[-1])
            from foodlab.collect import collect
            from foodlab.rag import history_text
            from foodlab.wiki import Wiki, WikiError
            notices = []
            with st.chat_message("assistant"):
                t0 = time.time(); steps = []
                def log(x):
                    steps.append(x); s.write(x)
                with st.status("관련 문서를 찾고 있습니다...", expanded=True) as s:
                    hist = history_text(msgs[:-1])
                    try:
                        wiki = Wiki(res["cfg"])
                        col_res = collect(res["cfg"], r, q, hist, res["chat"], res["emb"], wiki, res["ext_col"], progress=log)
                        if col_res["errors"]: notices.append("외부 문서 수집 중 일부 실패: " + "; ".join(col_res["errors"][:3]))
                        if col_res["selected"]: log("새로 색인한 문서: " + ", ".join(d["title"] for d in col_res["selected"]))
                        elif col_res["skipped_queries"] and not col_res["new_queries"]: log("이미 수집한 문서를 사용합니다")
                        elif col_res["new_queries"]: log("새로 색인할 만한 문서가 없었습니다 (후보가 음식·재료·조리법 문서가 아님)")
                        if col_res.get("item_self") == "NO_ITEM_SELF_DOC":
                            notices.append("이 레시피 자체를 다룬 외부 문서는 없어, 레시피 메타데이터와 원문 링크를 항목 근거로 사용했습니다.")
                    except WikiError as e:
                        notices.append(f"외부 문서 검색을 건너뛰었습니다: {e}")
                    except Exception as e:
                        notices.append(f"외부 문서 검색 실패 ({type(e).__name__}) — 레시피 메타데이터로만 답합니다.")
                    s.update(label="답변을 만드는 중..."); log("근거 검색 → 답변 생성 (프롬프트 3)")
                    try:
                        out = res["graph"].invoke({"recipe": r, "question": q, "history": hist})
                    except Exception as e:      # NF2004
                        print("[recipe] 답변 실패:", type(e).__name__)
                        out = {"answer": "답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.", "sources": [], "cited": [], "notes": ["FAILED"]}
                    summary = f"근거 {len(out.get('sources', []))}건 검색 · 인용 {len(out.get('cited', []))}건 · {time.time() - t0:.1f}초"
                    s.update(label=summary, state="complete", expanded=False)
                if "EXTERNAL_SEARCH_FAILED" in " ".join(out.get("notes", [])): notices.append("외부 문서 검색 단계에서 오류가 있었습니다.")
                if set(r["flags"]) & NUTRITION_FLAGS: notices.append("이 레시피의 원본 영양 수치에 이상치가 있어 검토가 필요합니다.")
            msgs.append({"role": "assistant", "content": out["answer"], "cited": out.get("cited", []), "notices": notices,
                         "steps": steps, "summary": summary})
            st.rerun()

# ---------------- 2. 내 재료로 찾기 (v0.2 유지 + 순위 보정) ----------------
elif page == P_ING:
    with st.container(border=True):
        owned = fridge_input("ing", label="🧊 가지고 있는 재료를 입력하세요")
        pantry = st.checkbox("물·소금·후추·식용유·설탕은 있다고 보기", value=False)
    with st.expander("필터·정렬", expanded=False):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        category = c1.selectbox("요리 종류", ["전체", "반찬", "일품", "후식", "밥", "국&찌개", "기타"])
        keyword = c2.text_input("메뉴 이름 검색", placeholder="예: 계란찜")
        sort_by = c3.selectbox("정렬", ["추천순", "부족 재료 적은 순", "보유 재료 많이 쓰는 순"])
        max_missing = c4.select_slider("부족 재료 허용", options=["제한 없음", 0, 1, 2, 3, 5], value="제한 없음")
    results = rank_recipes(recipes, owned, category, keyword, None if max_missing == "제한 없음" else max_missing, pantry, sort_by)
    if not owned:
        st.info("위 칸에 재료를 넣고 Enter 를 누르면 그 재료를 많이 쓰는 레시피 순으로 바뀝니다. 지금은 전체 레시피를 보여 줍니다.")
    else:
        st.caption(f"입력 재료 {len(owned)}개 · 결과 {len(results):,}개 / 전체 {len(recipes):,}개 — 추천 점수는 주요 재료 45% · 전체 재료 15% · "
                   "입력 재료 활용 40% 비중의 탐색용 지표이며 조리 가능성을 보장하지 않습니다.")
    if not results:
        st.warning("조건에 맞는 레시피가 없습니다.")
    else:
        limit = st.select_slider("표시 개수", options=[6, 12, 24, 48], value=12)
        shown = results[:limit]
        for i in range(0, len(shown), 3):
            for col, it in zip(st.columns(3, gap="medium"), shown[i:i + 3]):
                with col: recipe_card(it["recipe"], it if owned else None, key_prefix="ing")

# ---------------- 3. 조건으로 찾기 (F2011) ----------------
else:
    cond = st.text_input("🔎 원하는 조건", placeholder="예: 맵지 않고 두부가 들어간 반찬, 나트륨이 낮은 국")
    if not ai_ready:
        st.info("AI 기능이 준비되지 않아 조건 탐색을 쓸 수 없습니다. 위 안내를 확인하세요.")
    elif st.button("조건에 맞는 레시피 찾기", type="primary") and cond.strip():
        from foodlab.features import nl_search
        with st.status("레시피 색인에서 후보를 찾는 중...", expanded=False) as s:
            try:
                out = nl_search(res["cfg"], res["recipe_col"], res["emb"], res["chat"], cond.strip())
                s.update(label=f"조건을 만족하는 레시피 {out['candidates']}건 중 상위 {len(out['results'])}건", state="complete")
            except Exception as e:
                out = {"ok": False, "error": f"검색 실패 ({type(e).__name__})", "results": []}
                s.update(label="검색 실패", state="error")
        ss.nl_out = out
    out = ss.get("nl_out")
    if out:
        if out.get("error"): st.error(out["error"])
        if out.get("applied"): st.caption("적용한 확실한 조건: " + " · ".join(out["applied"]))
        if out.get("unsupported"):
            st.info("원본 데이터에 정보가 없어 판단하지 않은 조건: " + ", ".join(out["unsupported"]) + " (조리 시간·가격·난이도 정보는 공식 레시피 DB에 없습니다)", icon="ℹ️")
        for n in out.get("notes", []): st.caption("ⓘ " + n)
        if out["ok"] and not out["results"]: st.warning("조건에 맞는 레시피를 찾지 못했습니다. 조건을 바꿔 보세요.")
        for i in range(0, len(out["results"]), 3):
            for col, x in zip(st.columns(3, gap="medium"), out["results"][i:i + 3]):
                with col: recipe_card(R[x["rcp_seq"]], reason=x["reason"], key_prefix="nl")
        st.caption("요리 종류·영양 수치·재료 조건은 원본 메타데이터로 직접 거르고, '맵지 않은'처럼 수치로 못 거르는 조건만 "
                   "그 후보 안에서 AI가 재료를 보고 판단합니다. 후보에 없는 레시피는 표시하지 않습니다.")

# ---------------- 바닥글 ----------------
st.divider()
st.caption("⚕️ " + HEALTH)
st.caption("출처: 식품의약품안전처 식품안전나라 조리식품의 레시피 DB(COOKRCP01) · 한국어 위키백과(CC BY-SA 4.0)")
