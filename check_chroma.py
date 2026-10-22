"""ChromaDB 저장 위치 자체 점검 (외부 요청 없음, 임의 벡터 300개 사용 후 삭제)

    python check_chroma.py            # 실제 사용할 저장 위치(settings.chroma_path) 옆에서 시험
    python check_chroma.py --here     # 프로젝트 폴더(한글 경로) 안에서 시험 — 원인 확인용

색인 파일(HNSW)이 디스크에 써지고, 새 프로세스에서 다시 열리는지만 본다.
"""
import random
import shutil
import subprocess
import sys
from pathlib import Path

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DIM, N = 1536, 300
HNSW_FILES = ["header.bin", "data_level0.bin", "length.bin", "link_lists.bin"]


def write(path: str):
    import chromadb
    c = chromadb.PersistentClient(path=path)
    col = c.create_collection("selftest", metadata={"hnsw:space": "cosine", "hnsw:batch_size": 100, "hnsw:sync_threshold": 200})
    rnd = random.Random(0)
    for k in range(0, N, 64):
        ids = [f"t{i}" for i in range(k, min(k + 64, N))]
        col.add(ids=ids, embeddings=[[rnd.random() for _ in range(DIM)] for _ in ids], metadatas=[{"i": int(i[1:])} for i in ids])
    print(f"  쓰기 완료 {col.count()}건")


def read(path: str):
    import chromadb
    col = chromadb.PersistentClient(path=path).get_collection("selftest")
    got = col.get(include=["metadatas"])
    r = col.query(query_embeddings=[[0.5] * DIM], n_results=3)
    print(f"  다시 열기 OK: get {len(got['ids'])}건, query {r['ids'][0]}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] in ("_write", "_read"):
        (write if sys.argv[1] == "_write" else read)(sys.argv[2]); return
    from foodlab.settings import PROJECT, chroma_path, load_config
    base = PROJECT if "--here" in sys.argv else chroma_path(load_config()).parent
    t = base / "_chroma_selftest"
    shutil.rmtree(t, ignore_errors=True); t.mkdir(parents=True)
    print(f"[시험 위치] {t}")
    ok = True
    for step in ("_write", "_read"):
        p = subprocess.run([sys.executable, __file__, step, str(t)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(p.stdout.rstrip())
        if p.returncode != 0:
            ok = False; print("  실패:", (p.stderr.strip().splitlines() or ["?"])[-1]); break
    segs = [d for d in t.iterdir() if d.is_dir()]
    for d in segs:
        have = [f for f in HNSW_FILES if (d / f).exists()]
        print(f"  세그먼트 {d.name[:8]}… HNSW 파일 {len(have)}/{len(HNSW_FILES)} {have}")
        if len(have) < len(HNSW_FILES): ok = False
    if not segs: ok = False; print("  세그먼트 폴더가 없습니다.")
    shutil.rmtree(t, ignore_errors=True)
    print("[판정]", "정상 — 이 위치에 색인해도 됩니다." if ok else "실패 — 이 위치는 쓰면 안 됩니다.")


if __name__ == "__main__":
    main()
