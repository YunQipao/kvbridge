import sys
from types import ModuleType

import pytest

from kvbridge.formal import load_calibration_texts


@pytest.mark.parametrize("fail", [False, True])
def test_stream_iterator_closed_on_early_stop_or_error(monkeypatch, fail):
    closed = []

    class Stream:
        def shuffle(self, **kwargs):
            return self

        def __iter__(self):
            try:
                for i in range(20):
                    if fail and i == 1:
                        raise RuntimeError("read failed")
                    yield {"text": f"row {i}"}
            finally:
                closed.append(True)

    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: Stream()
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    if fail:
        with pytest.raises(RuntimeError, match="read failed"):
            load_calibration_texts(dataset="mock", n_calib=1)
    else:
        records = load_calibration_texts(dataset="mock", n_calib=1)
        assert records[0]["text"] == "row 0"
    assert closed == [True]
