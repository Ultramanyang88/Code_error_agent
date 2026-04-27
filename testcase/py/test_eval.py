import unittest
from eval_module import eval_expr


class TestEvalSimple(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(eval_expr("2+3"), 5)

    def test_subtraction(self):
        self.assertEqual(eval_expr("10-4"), 6)

    def test_multiplication(self):
        self.assertEqual(eval_expr("3*4"), 12)

    def test_division(self):
        self.assertEqual(eval_expr("20/5"), 4)

    def test_precedence(self):
        self.assertEqual(eval_expr("2+3*4"), 14)

if __name__ == "__main__":
    unittest.main()

