import pytest

from experiments.eval_mmlu_redux import _lm_eval_fewshot_rows


class NoTrainingTask:
    pass


def test_lm_eval_fewshot_refuses_test_rows():
    with pytest.raises(NotImplementedError):
        _lm_eval_fewshot_rows(NoTrainingTask(), 1)


class TrainingTask:
    def training_docs(self):
        return [{"question": "train", "choices": ["a", "b", "c", "d"],
                 "answer": 1}]

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_choice(self, doc):
        return doc["choices"]

    def doc_to_target(self, doc):
        return doc["answer"]


def test_lm_eval_fewshot_uses_training_docs():
    rows = _lm_eval_fewshot_rows(TrainingTask(), 1)
    assert rows[0]["prompt"] == "train"
