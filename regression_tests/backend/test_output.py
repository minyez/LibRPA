import io
import unittest

from regression_tests.backend.output import (
    ColorizedStatusStream,
    Tee,
    colorize_status_words,
)


class TestRegressionOutput(unittest.TestCase):

    def test_colorizes_pass_and_fail_words(self):
        text = colorize_status_words("PASS FAIL")

        self.assertEqual(text, "\033[32mPASS\033[0m \033[31mFAIL\033[0m")

    def test_tee_can_color_stdout_without_coloring_log(self):
        stdout = io.StringIO()
        log = io.StringIO()
        tee = Tee(ColorizedStatusStream(stdout, enable=True), log)

        tee.write("PASS FAIL\n")

        self.assertEqual(stdout.getvalue(),
                         "\033[32mPASS\033[0m \033[31mFAIL\033[0m\n")
        self.assertEqual(log.getvalue(), "PASS FAIL\n")


if __name__ == "__main__":
    unittest.main()
