import unittest
from src.vectordb.utils import preprocess_text


class TestUtils(unittest.TestCase):
    def test_preprocess_text(self):
        text = "Hello, World!\nThis is a test."
        expected_output = "Hello, World! This is a test."
        self.assertEqual(preprocess_text(text), expected_output)

        text = "This is a 'test' text."
        expected_output = "This is a 'test' text."
        self.assertEqual(preprocess_text(text, encoding=False), expected_output)

        text = "THIS IS A TEST."
        expected_output = "this is a test."
        self.assertEqual(preprocess_text(text, lowercase=True), expected_output)

        text = "Hello, World!  \nThis is a test."
        expected_output = "hello, world! this is a test."
        self.assertEqual(
            preprocess_text(text, encoding=True, lowercase=True, remove_newlines=True),
            expected_output,
        )


if __name__ == "__main__":
    unittest.main()