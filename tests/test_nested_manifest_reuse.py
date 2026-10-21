import hashlib
import json

import pytest
from kvbridge.protocol import reuse_manifest_prefix


def test_nested_manifest_prefixes_and_parent_provenance(tmp_path):
    records = [{"offset": i, "id": str(i), "text": f"row {i}"} for i in range(500)]
    parent = tmp_path / "calibration_manifest.json"
    parent.write_text(json.dumps({"records": records}, indent=2))
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    prefixes = {}
    for n in (50, 100, 200, 500):
        prefix = reuse_manifest_prefix(json.loads(parent.read_text()), n)
        prefixes[n] = prefix
        assert [r["offset"] for r in prefix] == list(range(n))
        assert len(prefix) == n
        assert parent_hash == hashlib.sha256(parent.read_bytes()).hexdigest()
    for small, large in ((50, 100), (100, 200), (200, 500)):
        assert prefixes[large][:small] == prefixes[small]
    with pytest.raises(ValueError, match="exceeds"):
        reuse_manifest_prefix(json.loads(parent.read_text()), 501)
