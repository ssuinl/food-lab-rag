"""GMS 채팅 모델 (LangChain ChatOpenAI). JSON 출력은 관대하게 파싱한다."""
from __future__ import annotations

import json
import re

from .settings import gms_key


class LLMError(RuntimeError):
    pass


def get_chat(cfg):
    from langchain_openai import ChatOpenAI
    key = gms_key()
    if not key:
        raise LLMError("GMS_KEY 가 .env 에 없습니다.")
    # gpt-5 계열은 temperature 지정을 거부할 수 있어 기본값을 쓴다.
    return ChatOpenAI(model=cfg["chat_model"], api_key=key, base_url=cfg["gms_base_url"], max_retries=2, timeout=90)


def ask(chat, system: str, user: str) -> str:
    try:
        r = chat.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    except Exception as e:  # 상세 오류는 화면에 노출하지 않는다(NF2004) — 호출부에서 일반 문구로 바꾼다
        raise LLMError(type(e).__name__) from None
    c = r.content
    if isinstance(c, list):
        c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def parse_json(text: str, kind=list):
    """```json ... ``` 이나 앞뒤 설명이 섞여도 첫 JSON 배열/객체를 꺼낸다. 실패하면 None."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    pat = r"\[.*\]" if kind is list else r"\{.*\}"
    m = re.search(pat, t, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, kind) else None
    except json.JSONDecodeError:
        return None
