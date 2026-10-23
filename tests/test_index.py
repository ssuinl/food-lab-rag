import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder

class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cfg["chroma_dir"] = str(Path(cls.tmp) / "chroma"); cfg["embedding_model"] = "fake-embed"
        p = Path(cls.tmp) / "config.json"; p.write_text(json.dumps(cfg), encoding="utf-8")
        os.environ["FOODLAB_CONFIG"] = str(p)

    def run_build(self, *args):
        import build_index
        with mock.patch("foodlab.embedder.GMSEmbedder", FakeEmbedder), mock.patch.object(sys, "argv", ["build_index.py", *args]):
            build_index.main()
        return json.loads((ROOT / "index_manifest.json").read_text(encoding="utf-8"))

    def test_idempotent_full_index(self):
        from foodlab.settings import load_config, find_raw_dir
        from foodlab.store import get_collection
        if not find_raw_dir(): self.skipTest("원본 raw 없음")
        m1 = self.run_build()
        self.assertEqual(m1["collection_count"], 1156); self.assertEqual(m1["embedded_this_run"], 1156)
        m2 = self.run_build()
        self.assertEqual(m2["collection_count"], 1156); self.assertEqual(m2["embedded_this_run"], 0)
        self.assertEqual(m2["requests_this_run"], 0)
        col = get_collection(load_config(), "recipe_meta", "fake-embed")
        got = col.get(ids=["recipe:28"], include=["metadatas", "documents"])
        self.assertEqual(got["metadatas"][0]["rcp_seq"], "28")
        self.assertIn("새우 두부 계란찜", got["documents"][0])
        self.assertIn("source_url", got["metadatas"][0])

    def test_model_mixing_refused(self):
        from foodlab.settings import load_config, find_raw_dir
        from foodlab.store import get_collection, StoreError
        if not find_raw_dir(): self.skipTest("원본 raw 없음")
        self.run_build("--limit", "5")
        with self.assertRaises(StoreError):
            get_collection(load_config(), "recipe_meta", "other-model")

if __name__ == "__main__":
    unittest.main()
