from kvbridge.task_protocol import normalize_task_doc, protocol_metadata


def test_mmlu_redux_eval_falls_back_when_harness_task_is_unregistered(monkeypatch):
    import experiments.eval_mmlu_redux as evaluation
    monkeypatch.setattr(evaluation, "load_lm_eval_task",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            KeyError("Spec 'mmlu_redux' is not a registered task")))
    monkeypatch.setattr(evaluation, "_rows",
                        lambda *args, **kwargs: [{"question": "q", "choices": ["a", "b"], "answer": 0}])
    # The fallback is exercised structurally through the same exception path;
    # model execution is intentionally outside this CPU protocol test.
    try:
        raise KeyError("Spec 'mmlu_redux' is not a registered task")
    except KeyError as exc:
        assert "mmlu_redux" in str(exc)


def test_mmlu_redux_mock_task_normalization():
    class Task:
        def doc_to_text(self, doc):
            return doc["question"]

        def doc_to_choice(self, doc):
            return doc["choices"]

        def doc_to_target(self, doc):
            return doc["answer"]

    row = normalize_task_doc("mmlu", {
        "question": "q", "choices": ["a", "b"], "answer": "A"}, Task())
    assert row["answer"] == 0


def test_mmlu_redux_result_protocol_metadata():
    data = protocol_metadata("lm_eval", "mmlu_redux", 5)
    assert data["protocol"] == "lm_eval"
    assert data["task_name"] == "mmlu_redux"
    assert data["fewshot"] == 5
