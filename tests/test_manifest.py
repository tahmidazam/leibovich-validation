import hashlib
import json

from leibovich_validation import manifest


def test_manifest_lists_files_with_checksums(tmp_path):
    data = tmp_path / "data"
    (data / "cohort").mkdir(parents=True)
    (data / "cohort" / "clinical.tsv").write_bytes(b"abc")
    (data / ".DS_Store").write_bytes(b"ignored")

    out = tmp_path / "manifest.json"
    assert manifest.write(data, out) == 1
    assert json.loads(out.read_text()) == {
        "cohort/clinical.tsv": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    }
