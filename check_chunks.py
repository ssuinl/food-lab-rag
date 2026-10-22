"""검증 4.2: 청크 크기·중복 길이를 바꾸면 청크 수가 달라지는지 확인 (외부 요청 없음)

    python check_chunks.py                      # 기본: 현재 설정 + 비교용 설정 2개
    python check_chunks.py 500:100 300:50       # 비교할 설정을 직접 지정 (청크크기:중복)

이미 수집해 둔 위키백과 본문(cache/wiki_pages)을 config.json 과 같은 분할기
(RecursiveCharacterTextSplitter)로 나눠서 문서별 청크 수를 표로 보여 준다.
실제 색인 설정은 config.json 의 chunk_size, chunk_overlap 이고, 여기서는 색인을 바꾸지 않는다.
"""
import json
import sys

from foodlab.collect import CACHE, chunk_text
from foodlab.settings import load_config

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def main():
    cfg = load_config()
    base = (int(cfg["chunk_size"]), int(cfg["chunk_overlap"]))
    args = [a for a in sys.argv[1:] if ":" in a]
    others = [tuple(int(x) for x in a.split(":")) for a in args] or [(500, 100), (2000, 200)]
    settings = [base] + [s for s in others if s != base]

    pages = []
    for f in sorted((CACHE / "wiki_pages").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("text"):
            pages.append(d)
    if not pages:
        raise SystemExit("cache/wiki_pages 에 수집된 문서가 없습니다. 앱이나 ask_cli.py 로 레시피 질문을 먼저 해 주세요.")

    head = " | ".join(f"{c}자/{o}자{' (현재)' if (c, o) == base else ''}" for c, o in settings)
    print(f"[4.2 청크 수 비교] 수집된 위키 문서 {len(pages)}개\n")
    print(f"| 문서 | 본문 길이 | {head} |")
    print("|---|---:|" + "---:|" * len(settings))
    totals = [0] * len(settings)
    for d in sorted(pages, key=lambda x: -len(x["text"])):
        counts = []
        for i, (c, o) in enumerate(settings):
            n = len(chunk_text({"chunk_size": c, "chunk_overlap": o}, d["text"]))
            counts.append(n); totals[i] += n
        print(f"| {d['title']} | {len(d['text']):,}자 | " + " | ".join(str(n) for n in counts) + " |")
    print("| **합계** | | " + " | ".join(f"**{t}**" for t in totals) + " |")
    print(f"\n설정은 config.json 의 chunk_size={base[0]}, chunk_overlap={base[1]} 에서 바꿉니다.")
    print("실제 색인에 반영하려면 설정을 바꾼 뒤 cache/ 와 external_docs 컬렉션을 비우고 다시 질문해야 합니다 (같은 검색어는 다시 수집하지 않기 때문).")


if __name__ == "__main__":
    main()
