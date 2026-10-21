from kvbridge.task_protocol import normalize_task_doc, format_winogrande
import sys
from types import ModuleType

import pytest

from kvbridge.task_protocol import load_lm_eval_task


def _install_task(monkeypatch, task):
    harness = ModuleType("lm_eval")
    tasks = ModuleType("lm_eval.tasks")
    tasks.get_task_dict = lambda names: {names[0]: task}
    harness.tasks = tasks
    monkeypatch.setitem(sys.modules, "lm_eval", harness)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks)


def test_harness_none_test_docs_falls_back_to_validation(monkeypatch):
    class Task:
        def test_docs(self):
            return None

        def validation_docs(self):
            return iter([{"label": "1"}])

    task = Task()
    _install_task(monkeypatch, task)
    loaded, docs = load_lm_eval_task("hellaswag")
    assert loaded is task
    assert docs == [{"label": "1"}]


def test_harness_respects_declared_split_availability(monkeypatch):
    class Task:
        def has_test_docs(self):
            return False

        def test_docs(self):
            raise AssertionError("unavailable split must not be loaded")

        def has_validation_docs(self):
            return True

        def validation_docs(self):
            return ["validation"]

    _install_task(monkeypatch, Task())
    assert load_lm_eval_task("hellaswag")[1] == ["validation"]


def test_explicit_missing_split_does_not_silently_fallback(monkeypatch):
    class Task:
        def test_docs(self):
            return None

        def validation_docs(self):
            return ["validation"]

    _install_task(monkeypatch, Task())
    with pytest.raises(ValueError, match="no test documents"):
        load_lm_eval_task("hellaswag", split="test")


class MockTask:
    def doc_to_text(self, doc):
        return doc["text"]

    def doc_to_choice(self, doc):
        return doc["choices"]

    def doc_to_target(self, doc):
        return doc["target"]


def test_winogrande_blank_replacement():
    prompt, choices, target = format_winogrande(
        {"sentence": "The _ ran.", "option1": "cat", "option2": "dog", "answer": "1"})
    assert prompt == "The {} ran."
    assert choices == ["cat", "dog"]
    assert target == 1


def test_harness_task_preprocessing_is_used():
    row = normalize_task_doc("mock", {"text": "P", "choices": ["a", "b"], "target": "B"},
                             MockTask())
    assert row["prompt"] == "P"
    assert row["choices"] == ["a", "b"]
    assert row["answer"] == 1


def test_arc_candidate_target_normalization():
    row = normalize_task_doc("arc_challenge", {
        "question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]},
        "answerKey": "B"})
    assert row["choices"] == ["x", "y"]
    assert row["answer"] == 1


def test_mmlu_answer_mapping():
    row = normalize_task_doc("mmlu", {"question": "q", "choices": ["a", "b"],
                                      "answer": "B"})
    assert row["answer"] == 1
