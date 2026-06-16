import re


_ANSI_RESET = "\033[0m"
_STATUS_COLORS = {
    "PASS": "\033[32m",
    "FAIL": "\033[31m",
}
_STATUS_RE = re.compile(r"\b(PASS|FAIL)\b")


def colorize_status_words(data: str) -> str:
    def repl(match):
        word = match.group(0)
        return "{}{}{}".format(_STATUS_COLORS[word], word, _ANSI_RESET)

    return _STATUS_RE.sub(repl, data)


class ColorizedStatusStream:
    def __init__(self, stream, enable=None):
        self._stream = stream
        if enable is None:
            enable = getattr(stream, "isatty", lambda: False)()
        self._enable = enable

    def write(self, data):
        if self._enable:
            data = colorize_status_words(data)
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()
