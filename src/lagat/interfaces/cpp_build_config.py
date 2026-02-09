from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def _find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find pyproject.toml from interface path.")


def _load_cpp_config(repo_root: Path) -> dict:
    config_path = repo_root / "pyproject.toml"
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("tool", {}).get("lagat", {}).get("cpp", {})


def resolve_cpp_lib_path(
    start: Path,
    interface_key: str,
    default_interface_dir: str,
    lib_filename: str,
) -> Path:
    repo_root = _find_repo_root(start)
    cfg = _load_cpp_config(repo_root)
    build_root = Path(cfg.get("build_root", "build"))
    interface_dir = Path(
        cfg.get(f"interface_{interface_key}_dir", default_interface_dir)
    )
    return repo_root / build_root / interface_dir / lib_filename


def ensure_lib_exists(lib_path: Path, interface_key: str) -> None:
    if lib_path.exists():
        return
    repo_root = _find_repo_root(lib_path)
    script = repo_root / "scripts/build_cpp.sh"
    subprocess.run([str(script), f"--{interface_key}"], check=True, cwd=repo_root)
    if lib_path.exists():
        return
    preset = f"interface-{interface_key}"
    raise RuntimeError(
        "C++ interface library not found at "
        f"{lib_path}. Auto-build failed. Try `{script} --{interface_key}` "
        "or "
        f"`cmake -S src/lagat/interfaces/{interface_key} --preset {preset} "
        f"&& cmake --build --preset {preset}`."
    )
