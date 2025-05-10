import unittest
from clickindex import DataIndexing
import os

class TestDataIndexing(unittest.TestCase):
    def setUp(self):
        self.indexer = DataIndexing()
        self.test_dir = "./test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "test.txt"), "w") as f:
            f.write("This is a test document.")

    def test_load_data(self):
        documents = self.indexer.load_data(self.test_dir)
        self.assertGreater(len(documents), 0)

    def test_get_embedding_obj(self):
        embedding = self.indexer.get_embedding_obj("huggingface")
        self.assertIsNotNone(embedding)
        with self.assertRaises(ValueError):
            self.indexer.get_embedding_obj("invalid")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()