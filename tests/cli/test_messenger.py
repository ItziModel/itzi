import pytest

import itzi.messenger as msgr


def test_fatal_raises_runtime_compatible_error(itzi_stderr):
    with pytest.raises(msgr.FatalError, match="expected failure") as error:
        msgr.fatal("expected failure")

    assert isinstance(error.value, RuntimeError)
    assert str(error.value) == "expected failure"
    assert itzi_stderr.getvalue().count("ERROR: expected failure") == 1
