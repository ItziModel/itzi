from io import StringIO

import pytest

import itzi.messenger as msgr


@pytest.fixture
def itzi_stderr(monkeypatch):
    console_handler = next(
        handler
        for handler in msgr._itzi_logger.logger.handlers
        if getattr(handler, "_itzi_console_handler", False)
    )
    stream = StringIO()
    monkeypatch.setattr(console_handler, "stream", stream)
    return stream
