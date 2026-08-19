import unittest
from eval_module import eval_expr, EvalError


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

    def test_division_by_zero_raises_eval_error(self):
        # Regression coverage for task_001_division_bug: the buggy fixture
        # (testcase/tasks/task_001_division_bug/buggy_files/eval_module.py)
        # removes the zero-check and lets a raw ZeroDivisionError escape
        # instead. None of the tests above exercise a zero divisor at all, so
        # without this one, task_001's eval always reported PASSED whether
        # the agent fixed the bug or not -- see the run_eval.py debugging
        # notes in README for the full story.
        with self.assertRaises(EvalError):
            eval_expr("5/0")
        # and specifically NOT a raw ZeroDivisionError leaking through
        try:
            eval_expr("5/0")
        except ZeroDivisionError:
            self.fail("eval_expr('5/0') raised ZeroDivisionError instead of EvalError")
        except EvalError:
            pass


class TestMainEntryPoint(unittest.TestCase):
    def test_main_module_imports_cleanly(self):
        # Regression coverage for task_003_import_error: the buggy fixture
        # (testcase/tasks/task_003_import_error/buggy_files/main.py) adds
        # `from nonexistent_module import helper` at module level. test_eval.py
        # only ever imported eval_module, never main -- so that bug was never
        # exercised by the test suite at all, and task_003's eval always
        # reported PASSED regardless of whether the agent fixed it.
        import main  # noqa: F401 -- import itself is the assertion; raises if main.py is broken


if __name__ == "__main__":
    unittest.main()
