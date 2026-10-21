"""Small adapter around lm-evaluation-harness task document semantics.

The mapped scorer only needs normalized prompt, choices and target.  Keeping
this boundary separate allows unit tests to use mock tasks without importing
or downloading the harness datasets.
"""

from __future__ import annotations


def _value(task, name, doc):
    value = getattr(task, name)
    return value(doc) if callable(value) else value


def normalize_target(target, choices):
    if isinstance(target, str):
        labels = {chr(ord("A") + i): i for i in range(len(choices))}
        if target in labels:
            return labels[target]
        if target.isdigit():
            return int(target)
    return int(target)


def format_winogrande(doc, choices=None, target=None):
    sentence = doc["sentence"]
    options = list(choices or [doc["option1"], doc["option2"]])
    if "_" not in sentence:
        raise ValueError("WinoGrande sentence has no blank")
    return sentence.replace("_", "{}"), options, normalize_target(
        target if target is not None else doc["answer"], options)


def normalize_task_doc(task_name, doc, task=None):
    name = task_name.lower()
    if task is not None:
        prompt = _value(task, "doc_to_text", doc)
        choices = _value(task, "doc_to_choice", doc)
        target = _value(task, "doc_to_target", doc)
        return {"prompt": str(prompt), "choices": list(choices),
                "answer": normalize_target(target, choices), "doc": doc}
    if "winogrande" in name:
        prompt, choices, answer = format_winogrande(doc)
        return {"prompt": prompt, "choices": choices, "answer": answer, "doc": doc}
    if "hellaswag" in name:
        return {"prompt": doc["ctx"], "choices": list(doc["endings"]),
                "answer": normalize_target(doc["label"], doc["endings"]), "doc": doc}
    if "arc" in name:
        labels = list(doc["choices"]["label"])
        choices = list(doc["choices"]["text"])
        answer = labels.index(str(doc["answerKey"]))
        return {"prompt": doc["question"], "choices": choices,
                "answer": answer, "doc": doc}
    choices = list(doc["choices"])
    return {"prompt": doc["question"], "choices": choices,
            "answer": normalize_target(doc["answer"], choices), "doc": doc}


def load_lm_eval_task(task_name, split=None):
    try:
        from lm_eval.tasks import get_task_dict
    except ImportError as exc:
        raise RuntimeError(
            "protocol=lm_eval requires the installed lm-evaluation-harness"
        ) from exc
    task_dict = get_task_dict([task_name])
    task = task_dict[task_name] if task_name in task_dict else next(iter(task_dict.values()))
    if split:
        docs = getattr(task, f"{split}_docs")()
    else:
        docs = None
        for candidate in ("test", "validation"):
            has_docs = getattr(task, f"has_{candidate}_docs", None)
            if has_docs is not None and not has_docs():
                continue
            get_docs = getattr(task, f"{candidate}_docs", None)
            if get_docs is not None:
                docs = get_docs()
                if docs is not None:
                    break
    if docs is None:
        requested = split or "test/validation"
        raise ValueError(f"lm_eval task {task_name!r} has no {requested} documents")
    return task, list(docs)


def protocol_metadata(protocol, task_name, fewshot):
    version = None
    if protocol == "lm_eval":
        try:
            import lm_eval
            version = getattr(lm_eval, "__version__", "unknown")
        except ImportError:
            version = None
    return {"protocol": protocol, "lm_eval_version": version,
            "task_name": task_name, "fewshot": int(fewshot)}
