"""F2001~F2002: 레시피 메타데이터 → ChromaDB 'recipe_meta' 컬렉션 (항목당 1청크, ID = recipe:{RCP_SEQ})

    python build_index.py --dry-run        # 네트워크 없음. 건수·예상 토큰·요청 수·미리보기만
    python build_index.py --limit 20       # 앞 20건만 색인 (시험용, 요청 1회)
    python build_index.py                  # 전체 색인. 바뀐/새 항목만 임베딩 (재실행 시 요청 0회)
    python build_index.py --verify         # 색인 건수 확인 + 검색 시험 3건 (질의 임베딩 3회)
    python build_index.py --rebuild        # 컬렉션 삭제 후 처음부터 (모델을 바꿨을 때만)

원칙
- 원본 raw/*.json 은 읽기만 한다. 실행 전후 SHA256 을 비교해 바뀌면 실패로 끝난다.
- 멱등성: 같은 ID 로 upsert, 내용 해시가 같으면 다시 임베딩하지 않는다. 원본에 없는 ID 는 지운다.
- 실패 복구: 배치마다 바로 저장하므로 중간에 멈춰도 다시 실행하면 남은 것만 이어서 처리한다.
"""
import argparse
import json
import sys
import time

from foodlab.data import EXPECTED_ROWS, load_recipes, sha256
from foodlab.item_text import TEXT_VERSION, build_metadata, build_text, content_hash
from foodlab.settings import PROJECT, find_raw_dir, load_config, safe_error

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

VERIFY_QUERIES = ["두부와 달걀로 만드는 반찬", "나트륨을 줄인 국이나 찌개", "단호박으로 만드는 디저트"]


def count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts), "tiktoken cl100k_base"
    except Exception:
        return int(sum(len(t) for t in texts) * 1.1), "문자 수 기반 추정"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config()
    raw = find_raw_dir()
    if not raw: raise SystemExit("원본 data_check/raw/cookrcp01_*.json 을 찾지 못했습니다.")
    before = {f.name: sha256(f) for f in sorted(raw.glob("cookrcp01_*.json"))}
    recipes, hashes = load_recipes(raw)
    print(f"[F2001] 원본 {raw} → 레시피 {len(recipes)}건 로드")
    if len(recipes) != EXPECTED_ROWS:
        raise SystemExit(f"레시피 수가 {EXPECTED_ROWS}건이 아닙니다 ({len(recipes)}). 중단합니다.")
    model = cfg.get("embedding_model") or "(미정)"
    items = []
    for r in recipes:
        text = build_text(r); md = build_metadata(r, hashes)
        md["content_hash"] = content_hash(text, md, model)
        items.append((f"recipe:{r['id']}", text, md))
    if a.limit: items = items[: a.limit]
    bs = cfg["embedding_batch_size"]

    if a.dry_run:
        tok, how = count_tokens([t for _, t, _ in items])
        lens = sorted(len(t) for _, t, _ in items)
        plan = {"items": len(items), "text_version": TEXT_VERSION, "estimated_tokens": tok, "token_method": how,
                "chars_min_median_max": [lens[0], lens[len(lens) // 2], lens[-1]],
                "batch_size": bs, "embedding_requests": -(-len(items) // bs), "network_calls": 0}
        (PROJECT / "index_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        (PROJECT / "index_preview.txt").write_text("\n\n----\n\n".join(t for _, t, _ in items[:3]), encoding="utf-8")
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        print("미리보기: index_preview.txt (앞 3건) / 계획: index_plan.json — 네트워크 요청 없음")
        return

    from foodlab.embedder import EmbedError, GMSEmbedder
    from foodlab.store import StoreError, drop_collection, get_collection
    try:
        emb = GMSEmbedder(cfg)
    except EmbedError as e:
        raise SystemExit(str(e))
    name = cfg["recipe_collection"]
    if a.rebuild:
        drop_collection(cfg, name); print(f"컬렉션 '{name}' 삭제 후 다시 만듭니다.")
    from foodlab.settings import chroma_path
    print(f"[저장 위치] {chroma_path(cfg)}")
    try:
        col = get_collection(cfg, name, emb.model, create=True)
    except StoreError as e:
        raise SystemExit(str(e))

    ids = [i for i, _, _ in items]
    have = col.get(include=["metadatas"])
    existing = {i: (m or {}).get("content_hash") for i, m in zip(have["ids"], have["metadatas"])}
    todo = [it for it in items if existing.get(it[0]) != it[2]["content_hash"]]
    stale = [] if a.limit else [i for i in existing if i.startswith("recipe:") and i not in set(ids)]
    if stale: col.delete(ids=stale)
    print(f"[F2002] 컬렉션 '{name}' 기존 {len(existing)}건 · 이번에 임베딩할 항목 {len(todo)}건 · 삭제 {len(stale)}건 · 요청 예정 {-(-len(todo)//bs)}회")
    t0 = time.time(); done = 0; failed = None
    for k in range(0, len(todo), bs):
        batch = todo[k:k + bs]
        try:
            vecs = emb.embed([t for _, t, _ in batch])
        except EmbedError as e:
            failed = str(e); break
        col.upsert(ids=[i for i, _, _ in batch], documents=[t for _, t, _ in batch],
                   metadatas=[m for _, _, m in batch], embeddings=vecs)
        done += len(batch)
        print(f"  {done}/{len(todo)} 저장 (요청 {emb.requests}회, 토큰 {emb.tokens})")
    total = col.count()
    hnsw_ok = _hnsw_files_ok(cfg, total)
    after = {f.name: sha256(f) for f in sorted(raw.glob("cookrcp01_*.json"))}
    if after != before: raise SystemExit("원본 파일 해시가 바뀌었습니다. 즉시 확인하세요.")
    manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "collection": name, "embedding_model": emb.model,
                "text_version": TEXT_VERSION, "collection_count": total, "expected": len(items),
                "embedded_this_run": done, "requests_this_run": emb.requests, "tokens_this_run": emb.tokens,
                "seconds": round(time.time() - t0, 1), "raw_sha256": before, "errors": emb.log, "failed": failed,
                "hnsw_files_ok": hnsw_ok}
    (PROJECT / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[결과] 컬렉션 건수 {total} / 기대 {len(items)} · 이번 임베딩 {done}건 · 요청 {emb.requests}회 · 토큰 {emb.tokens}")
    if hnsw_ok is False:
        raise SystemExit("[경고] 검색 색인 파일(HNSW)이 디스크에 써지지 않았습니다. 다음 실행에서 열리지 않습니다.\n"
                         "python check_chroma.py 로 저장 위치를 점검하세요.")
    if failed:
        raise SystemExit(f"중단: {failed}\n다시 실행하면 남은 {len(todo) - done}건만 이어서 처리합니다.")
    if a.verify:
        verify(cfg, col, emb)


def _hnsw_files_ok(cfg, total):
    """건수가 많아 HNSW 가 디스크에 저장됐어야 하는데 바이너리 파일이 없으면 False. 판단 불가면 None."""
    from foodlab.settings import chroma_path
    segs = [d for d in chroma_path(cfg).iterdir() if d.is_dir() and (d / "index_metadata.pickle").exists()]
    if not segs: return None
    return all((d / "data_level0.bin").exists() for d in segs)


def verify(cfg, col, emb):
    print("\n[검색 시험]")
    for q in VERIFY_QUERIES:
        try:
            v = emb.embed([q])[0]
        except Exception as e:
            print(f"  질의 임베딩 실패: {safe_error(e)}"); return
        r = col.query(query_embeddings=[v], n_results=3, include=["metadatas", "distances"])
        hits = [f"{m['display_name']}(#{m['rcp_seq']}, {m['category']}, 거리 {d:.3f})" for m, d in zip(r["metadatas"][0], r["distances"][0])]
        print(f"  '{q}' → " + " | ".join(hits))
    print(f"  (질의 임베딩 요청 {len(VERIFY_QUERIES)}회)")


if __name__ == "__main__":
    main()
