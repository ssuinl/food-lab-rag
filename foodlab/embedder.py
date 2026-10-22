"""GMS 임베딩 호출 (요청 수 상한·재시도·사용량 집계). 다른 유료 API로 자동 전환하지 않는다."""
from __future__ import annotations

import time

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

from .settings import gms_key, safe_error


class EmbedError(RuntimeError):
    pass


class GMSEmbedder:
    def __init__(self, cfg, model: str | None = None, max_requests: int | None = None):
        key = gms_key()
        if not key:
            raise EmbedError("GMS_KEY 가 .env 에 없습니다.")
        self.model = model or cfg.get("embedding_model")
        if not self.model:
            raise EmbedError("config.json 의 embedding_model 이 비어 있습니다. check_gms.py 로 확인한 모델명을 넣어 주세요.")
        self.client = OpenAI(api_key=key, base_url=cfg["gms_base_url"], max_retries=0, timeout=60)
        self.max_requests = max_requests if max_requests is not None else cfg.get("embedding_max_requests_per_run", 40)
        self.requests = 0
        self.tokens = 0
        self.log: list[dict] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        wait = 2.0
        for attempt in range(1, 4):
            if self.requests >= self.max_requests:
                raise EmbedError(f"이번 실행의 임베딩 요청 상한({self.max_requests}회)에 도달했습니다.")
            self.requests += 1
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts)
                if resp.usage is not None:
                    self.tokens += int(getattr(resp.usage, "total_tokens", 0) or getattr(resp.usage, "prompt_tokens", 0) or 0)
                vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
                if len(vecs) != len(texts):
                    raise EmbedError("응답 벡터 수가 입력 수와 다릅니다.")
                return vecs
            except RateLimitError as e:
                ra = None
                try: ra = float(e.response.headers.get("retry-after"))
                except Exception: pass
                self.log.append({"event": "429", "attempt": attempt}); time.sleep(min(ra or wait, 60)); wait *= 2
            except (APIConnectionError, APITimeoutError) as e:
                self.log.append({"event": "network", "attempt": attempt, "error": safe_error(e)}); time.sleep(wait); wait *= 2
            except APIStatusError as e:
                # 4xx(모델 미지원·권한 등)는 재시도해도 같으므로 즉시 중단
                self.log.append({"event": f"http_{e.status_code}", "error": safe_error(e)})
                if e.status_code >= 500: time.sleep(wait); wait *= 2; continue
                raise EmbedError(f"임베딩 요청 거부 (HTTP {e.status_code}): {safe_error(e)}") from None
        raise EmbedError("임베딩 요청이 3회 연속 실패했습니다.")
