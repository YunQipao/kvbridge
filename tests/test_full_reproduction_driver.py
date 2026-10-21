from pathlib import Path


def test_driver_has_paired_artifacts_and_stage_switches():
    script = (Path(__file__).resolve().parents[1] / "scripts/run_full_reproduction.sh").read_text()
    for value in ("FULL_HEAD_OUT", "CACHEBRIDGE_OUT", "--dry-run",
                  "--skip-fit", "--skip-paper-tasks", "--skip-sweep",
                  "--skip-attribution", "--skip-construction",
                  "--full-head-mapper", "--cachebridge-mapper",
                  "--prefix-lengths 128,512,1024"):
        assert value in script
