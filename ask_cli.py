"""터미널에서 RAG 한 번 돌려 보기 (검증 4.2·4.3·4.5 증빙용)

    python ask_cli.py --graph                                   # LangGraph 구조 출력 (retrieve → generate)
    python ask_cli.py 28 "두부 대신 쓸 수 있는 재료는?"          # 수집 과정·근거·답변 출력
    python ask_cli.py 28 "두부 대신 쓸 수 있는 재료는?" --no-collect     # 수집 없이 기존 색인만 사용
    python ask_cli.py 28 "두부 대신 쓸 수 있는 재료는?" --top-k 2        # Top-k 를 이번 실행에서만 바꿈
    python ask_cli.py 28 "두부 대신 쓸 수 있는 재료는?" --mmr            # MMR 검색으로 답변

    # 검증용 비교 (수집·답변 생성 없이 근거 목록만, 질의 임베딩만 사용)
    python ask_cli.py 28 "두부는 어떻게 만들어?" --compare-topk 1,2,4,8  # 4.3 Top-k 에 따른 근거 목록 변화
    python ask_cli.py 28 "두부는 어떻게 만들어?" --compare-mmr           # 4.5 F2014 유사도 vs MMR

외부 요청: 질문 1개당 LLM 3회(검색어·선별·답변) + 위키 최대 12회 + 임베딩(질의 1회 + 새 청크).
같은 레시피·같은 질문은 다시 수집하지 않는다. 비교 옵션은 LLM·위키 요청 없이 임베딩만 쓴다.
"""
import sys

from foodlab.collect import collect
from foodlab.data import load_recipes
from foodlab.embedder import GMSEmbedder
from foodlab.llm import get_chat
from foodlab.rag import build_graph, build_retrieve
from foodlab.settings import Config, find_raw_dir, load_config
from foodlab.store import get_collection
from foodlab.wiki import Wiki, WikiError

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def opt(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv): return sys.argv[i + 1]
    return default


def with_cfg(cfg, **kw):
    return Config({**cfg.data, **kw})


def show_sources(srcs, indent="  "):
    for s in srcs:
        extra = " · ".join(x for x in (s.get("relation") or "", f"청크 {s['id'].rsplit(':', 1)[-1]}" if s["type"] == "wikipedia" else "",
                                        f"거리 {s['distance']}" if "distance" in s else "") if x)
        print(f"{indent}[근거 {s['n']}] {s['label']} · {s['title']}" + (f" · {extra}" if extra else ""))


def compare_topk(cfg, rc, ec, emb, r, q, ks):
    print(f"\n[4.3 Top-k 비교] 질문: {q}  (설정 top_k_external = {cfg['top_k_external']})")
    for k in ks:
        res = build_retrieve(with_cfg(cfg, top_k_external=k, mmr=False), rc, ec, emb)({"recipe": r, "question": q})
        wiki = [s for s in res["sources"] if s["type"] == "wikipedia"]
        print(f"\n- Top-k = {k}: 근거 {len(res['sources'])}건 (레시피 메타데이터 1 + 위키 청크 {len(wiki)})")
        show_sources(res["sources"], "    ")


def compare_mmr(cfg, rc, ec, emb, r, q):
    k = int(cfg["top_k_external"])
    print(f"\n[F2014 유사도 vs MMR] 질문: {q}  (Top-k = {k}, MMR 은 후보 {k * 3}건에서 다양성 기준으로 {k}건 선택)")
    out = {}
    for name, mmr in (("유사도 검색", False), ("MMR 검색", True)):
        res = build_retrieve(with_cfg(cfg, mmr=mmr), rc, ec, emb)({"recipe": r, "question": q})
        wiki = [s for s in res["sources"] if s["type"] == "wikipedia"]
        out[name] = wiki
        docs = {}
        for s in wiki: docs[s["title"]] = docs.get(s["title"], 0) + 1
        print(f"\n- {name}: 위키 청크 {len(wiki)}건 · 서로 다른 문서 {len(docs)}개 ({', '.join(f'{t} {n}건' for t, n in docs.items())})")
        show_sources(res["sources"], "    ")
    a = {s["id"] for s in out["유사도 검색"]}; b = {s["id"] for s in out["MMR 검색"]}
    print(f"\n- 두 방식 공통 청크 {len(a & b)}건 · MMR 에만 있는 청크 {len(b - a)}건 · 유사도에만 있는 청크 {len(a - b)}건")
    if not a:
        print("  (이 레시피에 수집된 위키 청크가 없습니다. 먼저 --no-collect 없이 질문해서 문서를 모으세요.)")


def main():
    cfg = load_config()
    if opt("--top-k"): cfg = with_cfg(cfg, top_k_external=int(opt("--top-k")))
    if "--mmr" in sys.argv: cfg = with_cfg(cfg, mmr=True)
    emb = GMSEmbedder(cfg, max_requests=200)
    rc = get_collection(cfg, cfg["recipe_collection"], emb.model)
    if rc is None or rc.count() == 0: raise SystemExit("레시피 색인이 없습니다. python build_index.py 를 먼저 실행하세요.")
    ec = get_collection(cfg, cfg["external_collection"], emb.model, create=True)
    if "--graph" in sys.argv:
        print(build_graph(cfg, rc, ec, emb, get_chat(cfg)).get_graph().draw_mermaid()); return
    skip = {opt("--top-k"), opt("--compare-topk")}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in skip]
    if len(args) < 2: raise SystemExit(__doc__)
    seq, q = args[0], args[1]
    recipes, _ = load_recipes(find_raw_dir()); R = {r["id"]: r for r in recipes}
    if seq not in R: raise SystemExit(f"레시피 {seq} 가 없습니다.")
    r = R[seq]
    print(f"[레시피] {r['display_name']} #{seq}")
    print(f"[설정] chunk_size={cfg['chunk_size']} chunk_overlap={cfg['chunk_overlap']} top_k_external={cfg['top_k_external']} mmr={cfg.get('mmr')}")
    print(f"[색인] recipe_meta {rc.count():,}건 · external_docs {ec.count():,}건 (이 레시피 {len(ec.get(where={'rcp_seq': seq})['ids'])}건)")

    if opt("--compare-topk"):
        compare_topk(cfg, rc, ec, emb, r, q, [int(x) for x in opt("--compare-topk").split(",") if x.strip()])
        print(f"\n(이번 실행 임베딩 요청 {emb.requests}회, LLM·위키 요청 0회)"); return
    if "--compare-mmr" in sys.argv:
        compare_mmr(cfg, rc, ec, emb, r, q)
        print(f"\n(이번 실행 임베딩 요청 {emb.requests}회, LLM·위키 요청 0회)"); return

    chat = get_chat(cfg)
    g = build_graph(cfg, rc, ec, emb, chat)
    if "--no-collect" not in sys.argv:
        try:
            out = collect(cfg, r, q, "", chat, emb, Wiki(cfg), ec, progress=lambda m: print("  ·", m))
            print(f"[F2003 검색어] {out['queries']} ({out['query_method']}) / 새로 수집 {out['new_queries']} / 건너뜀 {out['skipped_queries']}")
            print("[F2004 선별] " + (", ".join(f"{d['title']}({d['relation']}, {d['selected_by']})" for d in out["selected"]) or "(새로 고른 문서 없음)"))
            print("[F2004 제외] " + (", ".join(f"{d['title']}({d['reason']})" for d in out["rejected"][:10]) or "(없음)"))
            print(f"[F2005 색인] 새 청크 {out['indexed_chunks']} · 임베딩 재사용 {out['reused_vectors']} · 항목 문서 상태 {out['item_self']}")
            if out["errors"]: print("[오류]", out["errors"])
        except WikiError as e:
            print("[수집 건너뜀]", e)
    res = g.invoke({"recipe": r, "question": q, "history": ""})
    print("\n[F2006 근거 목록]")
    for s in res["sources"]:
        extra = " · ".join(x for x in (s.get("relation") or "", f"청크 {s['id'].rsplit(':', 1)[-1]}" if s["type"] == "wikipedia" else "") if x)
        print(f"  [근거 {s['n']}] {s['label']} · {s['title']}" + (f" · {extra}" if extra else "") + f" · {s['url']}")
    print("\n[F2007 답변]\n" + res["answer"])
    print("\n[F2008 인용된 근거] " + (", ".join(f"[근거 {s['n']}] {s['title']}" for s in res["cited"]) or "인용 없음 — 화면에서는 '답변에 인용된 근거가 없습니다' 안내"))
    print(f"\n(이번 실행 임베딩 요청 {emb.requests}회, 메모 {res.get('notes')})")


if __name__ == "__main__":
    main()
