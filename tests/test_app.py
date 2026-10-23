"""Streamlit AppTest 로 화면 흐름 점검 (가짜 LLM·임베딩·위키, 네트워크 없음)."""
import json, os, shutil, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder, FakeChat, FakeWiki


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from foodlab.settings import find_raw_dir
        if not find_raw_dir(): raise unittest.SkipTest("원본 raw 없음")
        cls.tmp = tempfile.mkdtemp()
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cfg.update(chroma_dir=str(Path(cls.tmp) / "chroma"), embedding_model="fake-embed")
        p = Path(cls.tmp) / "config.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
        os.environ["FOODLAB_CONFIG"] = str(p); os.environ["GMS_KEY"] = "dummy"; os.environ["WIKI_CONTACT"] = "test@example.com"
        cls.patches = [mock.patch("foodlab.embedder.GMSEmbedder", FakeEmbedder), mock.patch("foodlab.llm.get_chat", lambda cfg: FakeChat()),
                       mock.patch("foodlab.wiki.Wiki", lambda cfg: FakeWiki()),
                       mock.patch("foodlab.collect.CACHE", Path(cls.tmp) / "cache")]
        for x in cls.patches: x.start()
        import build_index
        with mock.patch.object(sys, "argv", ["build_index.py"]): build_index.main()

    @classmethod
    def tearDownClass(cls):
        for x in cls.patches: x.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)

    def test_ingredient_page_and_chat(self):
        at = self.app(); at.run()
        self.assertFalse(at.exception, at.exception)
        at.radio(key="page").set_value("🥬 내 재료로 찾기").run()
        at.text_input(key="fridge_ing").input("달걀, 두부, 양파").run()
        self.assertEqual(at.session_state.fridge_text, "달걀, 두부, 양파")
        self.assertFalse(at.exception, at.exception)
        btn = [b for b in at.button if b.key and b.key.startswith("ing-")][0]
        btn.click().run()
        self.assertEqual(at.session_state.page, "💬 AI 셰프에게 묻기")
        self.assertEqual(at.text_input(key="fridge_recipe").value, "달걀, 두부, 양파")   # 재료는 화면 간 공유
        at.chat_input[0].set_value("두부 대신 뭘 써도 돼?").run()
        self.assertFalse(at.exception, at.exception)
        seq = at.session_state.selected_seq
        msgs = at.session_state.chats[seq]
        self.assertEqual(msgs[-1]["role"], "assistant")
        self.assertTrue(msgs[-1]["cited"])                          # 인용 근거 표시
        self.assertEqual(msgs[-1]["cited"][0]["type"], "recipe_meta")
        self.assertTrue(msgs[-1]["steps"]); self.assertIn("초", msgs[-1]["summary"])   # 처리 과정·소요 시간이 남음 (F2012)

    def test_general_chat_without_recipe(self):
        at = self.app(); at.run()
        self.assertEqual(at.session_state.page, "💬 AI 셰프에게 묻기")      # 메인 = 레시피 선택·재료 입력 없이 질문
        today = [b for b in at.button if b.key and b.key.startswith("today-")]
        self.assertEqual(len(today), 3)                                       # 오늘의 추천 메뉴 3개
        [b for b in at.button if b.label == "🔄 다른 추천"][0].click().run()
        self.assertFalse(at.exception, at.exception)
        at.text_input(key="fridge_general").input("두부, 달걀").run()       # 재료 입력 → 화면이 바로 바뀜
        self.assertEqual(len([b for b in at.button if b.key and b.key.startswith("mine-")]), 3)
        self.assertTrue(any("반영됨" in x.value for x in at.success))
        [b for b in at.button if b.key == "ex-0"][0].click().run()          # 예시 질문 버튼
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.session_state.chats["__general__"][-1]["role"], "assistant")
        at.chat_input[0].set_value("두부로 만들 반찬 추천해줘").run()
        self.assertFalse(at.exception, at.exception)
        m = at.session_state.chats["__general__"][-1]
        self.assertEqual(m["role"], "assistant"); self.assertTrue(m["cited"]); self.assertTrue(m["recipes"])
        btn = [b for b in at.button if b.key and b.key.startswith("g3-")][0]       # 근거 레시피 카드 → 레시피 대화
        btn.click().run()
        self.assertEqual(at.session_state.selected_seq, m["recipes"][0])
        [b for b in at.button if b.label == "← 전체 질문으로"][0].click().run()
        self.assertIsNone(at.session_state.selected_seq)
        at.radio(key="page").set_value("🔎 조건으로 찾기").run()
        at.button(key="home").click().run()                                 # 제목 → 첫 화면
        self.assertEqual(at.session_state.page, "💬 AI 셰프에게 묻기")
        self.assertTrue(at.session_state.chats["__general__"])               # 대화 기록 유지

    def test_nl_search(self):
        at = self.app(); at.run()
        at.radio(key="page").set_value("🔎 조건으로 찾기").run()
        at.text_input[0].input("두부 반찬").run()
        [b for b in at.button if b.label == "조건에 맞는 레시피 찾기"][0].click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(at.session_state.nl_out["ok"])

    def test_missing_key_notice(self):
        with mock.patch.dict(os.environ, {"GMS_KEY": ""}):
            at = self.app(); at.run()
            self.assertTrue(any("GMS_KEY" in w.value for w in at.warning))

if __name__ == "__main__":
    unittest.main()
