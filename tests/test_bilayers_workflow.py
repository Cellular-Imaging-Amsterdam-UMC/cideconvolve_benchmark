from pathlib import Path

from bilayers_cli import generate_cli_command, load_config, validate_config
from deconvolve import METHODS
from launcher import build_docker_command, build_local_command


ROOT = Path(__file__).resolve().parents[1]


def _defaults(config: dict) -> dict:
    return {item["name"]: item.get("default") for item in config["parameters"]}


def test_bilayers_config_is_valid_and_has_no_removed_dl_method() -> None:
    config = load_config(ROOT / "config.yaml")

    assert validate_config(config) == []
    assert "ci_rl_dl" not in METHODS
    assert "ci_rl_dl" not in (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "--infolder" in generate_cli_command(config)
    assert "--outfolder" in generate_cli_command(config)


def test_generic_launcher_builds_docker_and_local_commands() -> None:
    config = load_config(ROOT / "config.yaml")
    values = _defaults(config)

    docker = build_docker_command(config, values, "input", "output", gpu=True)
    local = build_local_command(
        config, values, "input", "output", python_executable="python"
    )

    assert docker[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "w_cideconvolve_benchmark:latest" in docker
    assert local[:2] == ["python", str(ROOT / "wrapper.py")]
    assert docker[-1] == "--local"
    assert local[-1] == "--local"
