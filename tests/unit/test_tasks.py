import subprocess
from unittest.mock import MagicMock, patch

from app.tasks import run_simulation


def test_run_simulation_fails_if_no_hostfile(tmp_path, monkeypatch):
    monkeypatch.setattr("app.tasks.HOSTFILE", tmp_path / "missing")

    res = run_simulation.apply(args=["dummy.yaml", 4]).get()

    assert res["status"] == "failed"
    assert "hostfile" in res["reason"]


@patch("subprocess.run")
def test_run_simulation_success(mock_run, tmp_path, monkeypatch):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("localhost slots=4\n")
    monkeypatch.setattr("app.tasks.HOSTFILE", hostfile)

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("dummy: yes")

    proc = MagicMock()
    proc.stdout = "OK"
    proc.stderr = ""
    proc.returncode = 0
    mock_run.return_value = proc

    res = run_simulation.apply(args=[str(yaml_path), 4]).get()

    assert res["status"] == "finished"
    assert res["stdout"] == "OK"
    assert res["stderr"] == ""


@patch("subprocess.run")
def test_run_simulation_subprocess_error(mock_run, tmp_path, monkeypatch):

    hostfile = tmp_path / "hostfile"
    hostfile.write_text("localhost slots=4\n")
    monkeypatch.setattr("app.tasks.HOSTFILE", hostfile)

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("dummy: yes")

    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=127,
        cmd=["mpirun", "..."],
        output="Some stdout output.",
        stderr="MPI command failed.",
    )

    res = run_simulation.apply(args=[str(yaml_path), 4]).get()

    assert res["status"] == "failed"
    assert "returncode" in res
    assert res["returncode"] == 127
    assert "stdout" in res
    assert "stderr" in res
    assert res["stderr"] == "MPI command failed."

    mock_run.assert_called_once()
