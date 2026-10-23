import json, os, shutil, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder, FakeChat, FakeWiki


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cfg.update(chroma_dir=str(Path(cls.tmp) / "chroma"), embedding_model="fake-embed", chunk_size=300, chunk_overlap=50)
        p = Path(cls.tmp) / "config.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
        os.environ["FOODLAB_CONFIG"] = str(p)
        import foodlab.collect as C
        cls.cache_patch = mock.patch.object(C, "CACHE", Path(cls.tmp) / "cache"); cls.cache_patch.start()
        from foodlab.settings import load_config, find_raw_dir
        from foodlab.data import load_recipes
        from foodlab.item_text import build_text, build_metadata
        from foodlab.store import get_collection
        if not find_raw_dir(): raise unittest.SkipTest("원본 raw 없음")
        cls.cfg = load_config(); cls.emb = FakeEmbedder(cls.cfg)
        recipes, hashes = load_recipes(find_raw_dir())
        cls.R = {r["id"]: r for r in recipes}
        cls.rcol = get_collection(cls.cfg, "recipe_meta", "fake-embed", create=True)
        sel = [cls.R[s] for s in ("28", "29", "31", "37", "1019")]
        cls.rcol.upsert(ids=[f"recipe:{r['id']}" for r in sel], documents=[build_text(r) for r in sel],
                        metadatas=[build_metadata(r, hashes) for r in sel], embeddings=cls.emb.embed([build_text(r) for r in sel]))
        cls.ecol = get_collection(cls.cfg, "external_docs", "fake-embed", create=True)

    @classmethod
    def tearDownClass(cls):
        cls.cache_patch.stop(); shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_collect_and_cache(self):
        from foodlab.collect import collect
        chat, wiki = FakeChat(), FakeWiki()
        r = collect(self.cfg, self.R["28"], "두부 대신 뭘 써도 돼?", "", chat, self.emb, wiki, self.ecol)
        self.assertEqual(r["queries"][0], "새우 두부 계란찜")                 # F2003 항목 자체 검색어 포함
        titles = {d["title"] for d in r["selected"]}
        self.assertIn("새우 두부 계란찜", titles)                              # 항목 자체 문서 강제 포함
        self.assertIn("두부", titles)
        self.assertNotIn("청주시", titles)                                     # 오선택 차단
        self.assertEqual(r["item_self"], "ITEM_SELF_DOC_FOUND")
        ids = self.ecol.get(where={"rcp_seq": "28"})["ids"]
        self.assertTrue(all(i.startswith("28:wiki-") and i.count(":") == 2 for i in ids))  # 항목ID:문서ID:순번
        n1 = self.ecol.count(); before = wiki.requests; llm_before = len(chat.calls)
        r2 = collect(self.cfg, self.R["28"], "두부 대신 뭘 써도 돼?", "", chat, self.emb, wiki, self.ecol)
        self.assertEqual(r2["new_queries"], []); self.assertEqual(wiki.requests, before)  # NF2002
        self.assertEqual(len(chat.calls), llm_before)                                    # 같은 질문 → 프롬프트1 재사용
        self.assertTrue(r2["query_method"].endswith("(cached)"))
        self.assertEqual(self.ecol.count(), n1)

    def test_2_other_item_reuses_vectors(self):
        from foodlab.collect import collect
        chat, wiki = FakeChat(), FakeWiki()
        r = collect(self.cfg, self.R["29"], "두부 보관법은?", "", chat, self.emb, wiki, self.ecol)
        self.assertGreater(r["reused_vectors"], 0)                         # 같은 문서 청크는 임베딩 재사용
        self.assertEqual(r["item_self"], "NO_ITEM_SELF_DOC")
        self.assertTrue(self.ecol.get(ids=["29:wiki-1:000"])["ids"])           # 다른 항목 ID 로 덮어쓰지 않음
        self.assertTrue(self.ecol.get(ids=["28:wiki-1:000"])["ids"])

    def test_3_graph_answer_cites_only_used(self):
        from foodlab.rag import build_graph
        g = build_graph(self.cfg, self.rcol, self.ecol, self.emb, FakeChat())
        out = g.invoke({"recipe": self.R["28"], "question": "두부 대신 뭘 써?", "history": ""})
        self.assertEqual(out["sources"][0]["type"], "recipe_meta")          # F2006 메타 항상 포함
        self.assertEqual([s["n"] for s in out["cited"]], [1, 2])             # F2008 인용된 것만
        self.assertTrue(len(out["sources"]) >= 2)
        self.assertEqual(len({s["id"] for s in out["sources"]}), len(out["sources"]))  # 중복 없음
        self.assertIn("retrieve", g.get_graph().nodes); self.assertIn("generate", g.get_graph().nodes)

    def test_4_no_meta_no_external(self):
        from foodlab.rag import build_graph
        fake_recipe = dict(self.R["1019"]); fake_recipe["id"] = "000000"
        g = build_graph(self.cfg, self.rcol, None, self.emb, FakeChat())
        out = g.invoke({"recipe": fake_recipe, "question": "?", "history": ""})
        self.assertIn("확인되지 않습니다", out["answer"]); self.assertEqual(out["cited"], [])

    def test_5_nl_search_rejects_unknown_ids(self):
        from foodlab.features import nl_search
        res = nl_search(self.cfg, self.rcol, self.emb, FakeChat(), "두부 반찬", n_candidates=5)
        self.assertTrue(res["ok"]); self.assertTrue(all(x["rcp_seq"] in self.R for x in res["results"]))
        self.assertNotIn("999999", [x["rcp_seq"] for x in res["results"]])
        self.assertEqual(res["unsupported"], ["조리 시간 15분"])                    # 판단 불가 조건은 화면에 알림
        self.assertTrue(res["results"])                                             # 판단 조건은 LLM 이 후보 안에서 선별
        self.assertTrue(any(a.startswith("AI 판단 조건") for a in res["applied"]))

    def test_5b_nl_search_hard_filters(self):
        from foodlab.features import nl_search
        cats = {r["category"] for r in (self.R[s] for s in ("28", "29", "31", "37", "1019"))}
        for cat in cats:
            chat = FakeChat(); chat.parsed = {"category": cat, "low_sodium": True, "judge": ["아무거나"], "search_text": "반찬"}
            res = nl_search(self.cfg, self.rcol, self.emb, chat, f"나트륨 적은 {cat}", n_candidates=5)
            self.assertTrue(all(x["meta"]["category"] == cat for x in res["results"]))        # 요리 종류는 코드로 강제
            self.assertTrue(any("나트륨" in a for a in res["applied"]))
            self.assertTrue(all(not x["meta"].get("nutrition_suspect") for x in res["results"]))

    def test_6_nutrition_flag_in_text(self):
        from foodlab.item_text import build_text
        self.assertIn("이상치", build_text(self.R["1019"]))

if __name__ == "__main__":
    unittest.main()
