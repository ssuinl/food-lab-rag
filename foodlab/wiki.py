"""한국어 위키백과 API (검색·본문). User-Agent 연락처 필수, 429 Retry-After, 호출 간격, 요청 상한 (NF2006)."""
from __future__ import annotations

import html
import re
import time

import requests

from .settings import wiki_contact

API = "https://ko.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0 (한국어 위키백과)"


class WikiError(RuntimeError):
    pass


class Wiki:
    def __init__(self, cfg, max_requests: int | None = None):
        c = wiki_contact()
        if not c:
            raise WikiError("WIKI_CONTACT 가 .env 에 없습니다 (위키백과 User-Agent 정책).")
        self.ua = f"FoodLab/0.3 ({c}) python-requests"
        try:
            self.ua.encode("latin-1")
        except UnicodeEncodeError:
            raise WikiError("WIKI_CONTACT 에 한글이 들어 있습니다.") from None
        self.interval = cfg.get("request_interval_sec", 0.5)
        self.max_requests = max_requests or cfg.get("wiki_max_requests_per_question", 12)
        self.requests = 0
        self.log: list[dict] = []

    def _get(self, params: dict) -> dict:
        wait = 2.0
        for attempt in range(1, 4):
            if self.requests >= self.max_requests:
                raise WikiError(f"질문당 위키 요청 상한({self.max_requests}회)에 도달했습니다.")
            self.requests += 1
            try:
                r = requests.get(API, params={**params, "format": "json", "formatversion": 2},
                                 headers={"User-Agent": self.ua}, timeout=20)
            except requests.RequestException as e:
                self.log.append({"event": "network", "error": type(e).__name__}); time.sleep(wait); wait *= 2; continue
            if r.status_code == 429:
                ra = min(float(r.headers.get("Retry-After") or wait), 60)
                self.log.append({"event": "429", "wait": ra}); time.sleep(ra); wait *= 2; continue
            if r.status_code >= 500:
                self.log.append({"event": f"http_{r.status_code}"}); time.sleep(wait); wait *= 2; continue
            if r.status_code != 200:
                raise WikiError(f"위키백과 응답 오류 HTTP {r.status_code}")
            time.sleep(self.interval)
            return r.json()
        raise WikiError("위키백과 요청이 3회 실패했습니다.")

    def search(self, query: str, limit: int = 5) -> list[dict]:
        d = self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
        return [{"page_id": x["pageid"], "title": x["title"],
                 "snippet": html.unescape(re.sub(r"<[^>]+>", "", x.get("snippet", "")))}
                for x in d.get("query", {}).get("search", [])]

    def fetch(self, title: str) -> dict | None:
        d = self._get({"action": "query", "prop": "extracts|info|pageprops", "titles": title, "explaintext": 1,
                       "inprop": "url", "redirects": 1, "ppprop": "disambiguation"})
        pages = d.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or pages[0].get("invalid"):
            return None
        p = pages[0]
        return {"page_id": p["pageid"], "title": p["title"], "url": p.get("fullurl"),
                "revision": p.get("lastrevid"), "disambiguation": "disambiguation" in (p.get("pageprops") or {}),
                "text": p.get("extract", "")}
