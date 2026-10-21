"""경로·설정·키 로딩. 키 값은 절대 출력하지 않는다 (NF2003, NF2004)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent.parent          # food_lab_v0_3/
LAB = PROJECT.parent                                         # streamlit_lab/


def _load_env() -> None:
    # 프로젝트 폴더의 .env 를 먼저, 없으면 상위(streamlit_lab)의 .env 를 읽는다.
    for p in (PROJECT / ".env", LAB / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


_load_env()


@dataclass(frozen=True)
class Config:
    data: dict

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)


def load_config(path: Path | None = None) -> Config:
    p = path or Path(os.getenv("FOODLAB_CONFIG") or PROJECT / "config.json")
    return Config(json.loads(p.read_text(encoding="utf-8")))


def gms_key() -> str | None:
    k = os.getenv("GMS_KEY", "").strip()
    return k or None


def wiki_contact() -> str | None:
    c = os.getenv("WIKI_CONTACT", "").strip()
    return c or None


def find_raw_dir() -> Path | None:
    """원본 COOKRCP01 위치 (읽기 전용)."""
    for base in (LAB / "data_check", PROJECT / "data_check", Path.cwd() / "data_check"):
        raw = base / "raw"
        if raw.is_dir() and list(raw.glob("cookrcp01_*.json")):
            return raw
    return None


def _ascii(p: Path) -> bool:
    try:
        str(p).encode("ascii"); return True
    except UnicodeEncodeError:
        return False


def chroma_path(cfg: Config) -> Path:
    """ChromaDB 저장 위치.
    Windows 에서 경로에 한글 등 비ASCII 문자가 있으면 HNSW 색인 파일(C++ 로더)을 다시 열지 못한다
    ("Error loading hnsw index"). 그래서 Windows + 비ASCII 경로면 ASCII 경로로 옮긴다.
    우선순위: 환경변수 FOODLAB_CHROMA_DIR > config 의 절대경로 > 프로젝트 폴더 기준 상대경로."""
    env = os.getenv("FOODLAB_CHROMA_DIR", "").strip()
    if env:
        return Path(env)
    p = Path(cfg["chroma_dir"])
    p = p if p.is_absolute() else PROJECT / p
    if os.name == "nt" and not _ascii(p):
        home = Path.home()
        base = home if _ascii(home) else Path((os.getenv("SystemDrive") or "C:") + "\\")
        p = base / "foodlab_v03" / Path(cfg["chroma_dir"]).name
    return p


def safe_error(e: Exception) -> str:
    """오류 문구에 키가 섞여 나오지 않게 가린다."""
    s = f"{type(e).__name__}: {e}"
    k = gms_key()
    if k:
        s = s.replace(k, "***")
    return s[:300]
