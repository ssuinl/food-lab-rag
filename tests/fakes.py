"""테스트용 가짜 임베딩 (네트워크 없음). 문자 n-gram 해시 벡터 — 실제 모델 품질과 무관."""
import hashlib, math

class FakeEmbedder:
    def __init__(self, cfg, model=None, max_requests=None):
        self.model = model or "fake-embed"; self.requests = 0; self.tokens = 0; self.log = []
        self.max_requests = max_requests or 10_000
    def embed(self, texts):
        self.requests += 1
        out = []
        for t in texts:
            v = [0.0] * 64
            s = t.replace(" ", "")
            for i in range(len(s) - 1):
                h = int(hashlib.md5(s[i:i+2].encode()).hexdigest(), 16)
                v[h % 64] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class _Msg:
    def __init__(self, c): self.content = c


class FakeChat:
    """프롬프트 종류별로 정해진 응답을 돌려주는 가짜 LLM."""
    def __init__(self): self.calls = []
    def invoke(self, messages):
        sys_, user = messages[0]["content"], messages[1]["content"]
        self.calls.append(sys_[:20])
        if "검색어를 정한다" in sys_:
            return _Msg('["두부", "달걀"]')
        if "편집자" in sys_:
            import re, json
            titles = re.findall(r"^- ([^:]+):", user, re.M)
            pick = [{"title": t, "relation": "INGREDIENT_BACKGROUND", "reason": "재료 설명"} for t in titles if t in ("두부", "달걀")]
            return _Msg(json.dumps(pick, ensure_ascii=False))
        if "푸드랩 AI 셰프" in sys_:
            return _Msg("연두부와 달걀을 섞어 찝니다 [근거 1]. 두부는 콩으로 만듭니다 [근거 2].")
        if "조건 해석기" in sys_:
            import json
            return _Msg(json.dumps(getattr(self, "parsed", {"category": None, "judge": ["두부가 들어간"], "unsupported": ["조리 시간 15분"],
                                                           "search_text": "두부 반찬"}), ensure_ascii=False))
        if "레시피 검색 도우미" in sys_ or "레시피 검토자" in sys_:
            import re, json
            seqs = re.findall(r"^- (\d+) \|", user, re.M)[:2]
            return _Msg(json.dumps([{"rcp_seq": s, "reason": "조건 일치"} for s in seqs] + [{"rcp_seq": "999999", "reason": "가짜"}]))
        return _Msg("[]")


class FakeWiki:
    def __init__(self): self.requests = 0; self.log = []
    PAGES = {"두부": "두부는 콩을 갈아 만든 식품이다. " * 80, "달걀": "달걀은 닭의 알이다. " * 60,
             "새우 두부 계란찜": "새우 두부 계란찜 설명. " * 10}
    def search(self, q, limit=5):
        self.requests += 1
        base = [{"page_id": 1, "title": "두부", "snippet": "콩으로 만든 식품"}, {"page_id": 2, "title": "달걀", "snippet": "닭의 알"},
                {"page_id": 3, "title": "청주시", "snippet": "충청북도의 시"}]
        if "계란찜" in q: base.append({"page_id": 9, "title": "새우 두부 계란찜", "snippet": "음식"})
        return base[:limit]
    def fetch(self, title):
        self.requests += 1
        pid = {"두부": 1, "달걀": 2, "새우 두부 계란찜": 9}.get(title)
        if not pid: return None
        return {"page_id": pid, "title": title, "url": f"https://ko.wikipedia.org/wiki/{title}", "revision": 1,
                "disambiguation": False, "text": self.PAGES[title]}
