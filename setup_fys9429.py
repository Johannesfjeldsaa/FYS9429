#!/usr/bin/env python3
"""
FYS9429 Python Environment Setup.

Creates a conda environment with all dependencies for the FYS9429 course,
including xesmf/esmpy from conda-forge, Ray[tune], and editable installs
of the local FYS9429 package (via pyproject.toml), METEOR (tag v1.6.0),
and general_backend.

PyTorch, CUDA, cuDNN, and torchvision are NOT installed into the conda
environment. Instead they are provided by the cluster's HPC module system
(EasyBuild/Lmod) to avoid large downloads and wasted disk quota:
    CUDA/12.1.1
    cuDNN/8.9.2.26-CUDA-12.1.1
    PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    torchvision/0.16.0-foss-2023a-CUDA-12.1.1

The script automatically adds 'module load' commands to .envrc (direnv)
and configures VS Code to open a login shell so the modules are always
available when you open the project.

Before running, load Miniforge3 so that 'conda' is on PATH:
    module load Miniforge3/24.11.3-0

Usage:
    python setup_fys9429.py --prefix /path/to/conda/env
    python setup_fys9429.py --name fys9429 --register-kernel
    python setup_fys9429.py --prefix /path/to/env --force  # recreate
    python setup_fys9429.py --no-system-modules            # skip module integration
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# HPC system modules (provided by the cluster, not installed into conda)
# ---------------------------------------------------------------------------
# Loaded at runtime via 'module load'; kept out of the conda env to
# avoid downloading large GPU packages unnecessarily.
_SYSTEM_MODULES: list[str] = [
    "CUDA/12.1.1",
    "cuDNN/8.9.2.26-CUDA-12.1.1",
    "PyTorch/2.1.2-foss-2023a-CUDA-12.1.1",
    "torchvision/0.16.0-foss-2023a-CUDA-12.1.1",
]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def clone_or_update_repo(
    dest: Path,
    ssh_url: str,
    https_url: str,
    ref: str = "main",
    is_tag: bool = False,
) -> tuple[bool, str]:
    """
    Ensure a repository is available at *dest* checked out at *ref*.

    For branches, fetches and fast-forwards. For tags, fetches and
    checks out the tag (no pull). Returns (success, used_url).
    Tries SSH first, falls back to HTTPS.
    """
    dest = dest.resolve()
    git_cmd = shutil.which("git")
    if git_cmd is None:
        raise RuntimeError("git is not available on PATH; cannot clone repos.")

    if dest.exists():
        if (dest / ".git").exists():
            print(
                f"Repository {dest.name} found at {dest} — "
                f"fetching and checking out {ref!r}."
            )
            try:
                subprocess.run(
                    [git_cmd, "-C", str(dest), "fetch", "--tags"], check=True
                )
                subprocess.run(
                    [git_cmd, "-C", str(dest), "checkout", ref], check=True
                )
                if not is_tag:
                    subprocess.run(
                        [git_cmd, "-C", str(dest), "pull", "--ff-only"],
                        check=True,
                    )
                return True, "existing"
            except subprocess.CalledProcessError as exc:
                print(
                    f"Warning: failed to update {dest.name}: {exc}. "
                    "Consider deleting the directory and retrying.",
                    file=sys.stderr,
                )
                return False, "existing-failed"
        raise RuntimeError(f"{dest} exists but is not a git repository.")

    tag_or_branch = "--branch"  # git clone accepts both branch names and tags
    print(f"Cloning {dest.name} into {dest} (trying SSH first)…")
    try:
        subprocess.run(
            [git_cmd, "clone", tag_or_branch, ref, ssh_url, str(dest)],
            check=True,
        )
        return True, ssh_url
    except subprocess.CalledProcessError as exc:
        print(f"SSH clone failed: {exc}", file=sys.stderr)
        print("Falling back to HTTPS clone…")
        try:
            subprocess.run(
                [git_cmd, "clone", tag_or_branch, ref, https_url, str(dest)],
                check=True,
            )
            return True, https_url
        except subprocess.CalledProcessError as exc2:
            raise RuntimeError(
                f"Both SSH and HTTPS clone attempts for {dest.name} failed. "
                "Ensure network access and credentials are working."
            ) from exc2


# ---------------------------------------------------------------------------
# Conda helpers
# ---------------------------------------------------------------------------

def _conda_target_args(
    env_name: str | None,
    env_prefix: Path | None,
) -> list[str]:
    if env_prefix is not None:
        return ["--prefix", str(env_prefix)]
    if env_name is not None:
        return ["--name", env_name]
    raise ValueError("Either env_name or env_prefix must be provided.")


def _sanitized_env() -> dict[str, str]:
    """Return os.environ with virtualenv / conda noise removed."""
    run_env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    ):
        run_env.pop(key, None)
    current_path = run_env.get("PATH", "")
    run_env["PATH"] = ":".join(
        p for p in current_path.split(":") if p and "/.virtualenvs/" not in p
    )
    run_env["PYTHONNOUSERSITE"] = "1"
    return run_env


def _run_conda(
    conda_cmd: str,
    args: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [conda_cmd, *args],
        check=True,
        text=True,
        capture_output=capture_output,
        env=_sanitized_env(),
    )


def _conda_env_exists(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
) -> bool:
    result = _run_conda(conda_cmd, ["env", "list", "--json"], capture_output=True)
    envs = {Path(p).resolve() for p in json.loads(result.stdout).get("envs", [])}
    if env_prefix is not None:
        return env_prefix.resolve() in envs
    return any(p.name == env_name for p in envs)


def _run_in_env(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
    command: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    target = _conda_target_args(env_name, env_prefix)
    return subprocess.run(
        [conda_cmd, "run", *target, *command],
        check=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        capture_output=capture_output,
        env=_sanitized_env(),
    )


def _get_env_prefix(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
) -> Path:
    if env_prefix is not None:
        return env_prefix.resolve()
    cp = _run_in_env(
        conda_cmd,
        env_name,
        env_prefix,
        ["python", "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
    )
    return Path(cp.stdout.strip()).resolve()


def _get_site_packages(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
) -> str:
    code = (
        "import site; "
        "paths=[p for p in site.getsitepackages() if 'site-packages' in p]; "
        "print(paths[0] if paths else '')"
    )
    cp = _run_in_env(
        conda_cmd, env_name, env_prefix, ["python", "-c", code], capture_output=True
    )
    site_packages = cp.stdout.strip()
    if not site_packages:
        raise RuntimeError("Failed to detect site-packages path inside conda env.")
    return site_packages


def _ensure_esmf_module_alias(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
) -> None:
    """Create an ESMF.py shim so older code that imports ESMF still works."""
    code = (
        "import importlib.util, pathlib, site\n"
        "esmf_spec = importlib.util.find_spec('ESMF')\n"
        "esmpy_spec = importlib.util.find_spec('esmpy')\n"
        "if esmf_spec is None and esmpy_spec is not None:\n"
        "    site_pkgs = [p for p in site.getsitepackages() if 'site-packages' in p]\n"
        "    target = pathlib.Path(site_pkgs[0]) / 'ESMF.py'\n"
        "    if not target.exists():\n"
        "        target.write_text('from esmpy import *\\n')\n"
        "        print(f'Created ESMF compatibility shim: {target}')\n"
    )
    _run_in_env(conda_cmd, env_name, env_prefix, ["python", "-c", code])


# ---------------------------------------------------------------------------
# HPC module system helpers
# ---------------------------------------------------------------------------

def _find_lmod_init() -> str | None:
    """Return the path to the Lmod bash init script, or None."""
    # $LMOD_PKG is set by Lmod itself when the module system is initialised.
    lmod_pkg = os.environ.get("LMOD_PKG")
    if lmod_pkg:
        candidate = Path(lmod_pkg) / "init" / "bash"
        if candidate.exists():
            return str(candidate)
    for path in [
        "/etc/profile.d/lmod.sh",
        "/usr/share/lmod/lmod/init/bash",
    ]:
        if Path(path).exists():
            return path
    return None


def _modules_available() -> bool:
    """Return True if the Lmod module system is initialised in this session."""
    return bool(
        os.environ.get("LMOD_CMD")
        or os.environ.get("LMOD_PKG")
        or shutil.which("modulecmd")
    )


def _run_in_env_with_modules(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
    command: list[str],
    modules: list[str],
    lmod_init: str,
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *command* inside the conda env after loading HPC *modules*."""
    target = _conda_target_args(env_name, env_prefix)
    conda_run_args = [conda_cmd, "run", *target, *command]
    module_loads = " && ".join(f"module load {m}" for m in modules)
    bash_cmd = (
        f"source {shlex.quote(lmod_init)} 2>/dev/null"
        f" && {module_loads}"
        f" && {' '.join(shlex.quote(a) for a in conda_run_args)}"
    )
    return subprocess.run(
        ["bash", "-c", bash_cmd],
        check=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=_sanitized_env(),
    )


# ---------------------------------------------------------------------------
# Repo-specific install helpers
# ---------------------------------------------------------------------------

def _install_repo(
    conda_cmd: str,
    env_name: str | None,
    env_prefix: Path | None,
    dest: Path,
    label: str,
    extra: str = "",
) -> bool:
    """pip-install *dest* as an editable package; return True on success."""
    install_path = str(dest) + (f"[{extra}]" if extra else "")
    try:
        _run_in_env(
            conda_cmd,
            env_name,
            env_prefix,
            ["python", "-m", "pip", "install", "-e", install_path],
        )
        print(f"Successfully installed {label} (editable).")
        return True
    except subprocess.CalledProcessError:
        print(
            f"Editable install of {label} failed (see pip output above). "
            "The repo is still available on disk.",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup FYS9429 Python environment with conda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Remember to load the conda module first:\n"
            "  module load Miniforge3/24.11.3-0"
        ),
    )
    parser.add_argument("--prefix", "-p", default=None,
                        help="Full path to conda env prefix")
    parser.add_argument("--name", "-n", default="fys9429",
                        help="Conda environment name (ignored when --prefix is given)")
    parser.add_argument("--project-root", "-r", default=".",
                        help="Project root directory (default: current directory)")
    parser.add_argument("--env-file", default="environment.yml",
                        help="Conda environment YAML file (default: environment.yml)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Remove and recreate the environment if it already exists")
    parser.add_argument("--no-vscode", action="store_true",
                        help="Skip writing .vscode/settings.json")
    parser.add_argument("--register-kernel", action="store_true",
                        help="Register the env as a Jupyter kernel for the current user")
    parser.add_argument("--kernel-name", default="fys9429",
                        help="Jupyter kernel name (default: fys9429)")
    parser.add_argument("--kernel-display", default="FYS9429 (py3.12)",
                        help="Jupyter kernel display name")
    parser.add_argument("--backend-branch", default="main",
                        help="Branch to check out for general_backend (default: main)")
    parser.add_argument("--conda-cmd", default=None,
                        help="Path to the conda executable (auto-detected if omitted)")
    parser.add_argument("--no-system-modules", action="store_true",
                        help="Skip HPC system module integration (do not add module loads "
                             "to .envrc or VS Code settings)")
    args = parser.parse_args()

    # ---- Resolve project root ------------------------------------------------
    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)
    env_file = (project_root / args.env_file).resolve()

    if not env_file.exists():
        print(f"ERROR: environment file not found: {env_file}", file=sys.stderr)
        return 10

    # ---- Resolve env target --------------------------------------------------
    if args.prefix:
        env_prefix = Path(args.prefix).expanduser().resolve()
        env_name = None
    else:
        env_name = args.name
        env_prefix = None

    # ---- Find conda ----------------------------------------------------------
    conda_cmd = args.conda_cmd or shutil.which("conda")
    if conda_cmd is None:
        print(
            "ERROR: 'conda' not found on PATH.\n"
            "Load the module first:\n"
            "  module load Miniforge3/24.11.3-0",
            file=sys.stderr,
        )
        return 2

    # ---- Summary -------------------------------------------------------------
    print(f"Project root : {project_root}")
    print(f"Env file     : {env_file}")
    print(f"Conda cmd    : {conda_cmd}")
    if env_prefix is not None:
        print(f"Env prefix   : {env_prefix}")
    else:
        print(f"Env name     : {env_name}")

    use_system_modules = not args.no_system_modules and _modules_available()
    lmod_init = _find_lmod_init() if use_system_modules else None
    if use_system_modules:
        if lmod_init:
            print(f"Lmod init    : {lmod_init}")
        else:
            print("WARNING: Lmod init script not found — module loads will be skipped.")
            use_system_modules = False
    print(f"System mods  : {'yes (' + ', '.join(_SYSTEM_MODULES) + ')' if use_system_modules else 'disabled / not detected'}")
    print()

    # ---- Create / update conda env -------------------------------------------
    env_exists = _conda_env_exists(conda_cmd, env_name, env_prefix)
    if env_exists and args.force:
        print("Removing existing conda env (--force)…")
        _run_conda(
            conda_cmd,
            ["env", "remove", "-y", *_conda_target_args(env_name, env_prefix)],
        )
        env_exists = False

    if not env_exists:
        print("Creating conda environment from YAML…")
        _run_conda(
            conda_cmd,
            [
                "env", "create", "-y",
                "--file", str(env_file),
                *_conda_target_args(env_name, env_prefix),
            ],
        )
    else:
        print("Updating existing conda environment from YAML…")
        _run_conda(
            conda_cmd,
            [
                "env", "update",
                "--file", str(env_file),
                "--prune",
                *_conda_target_args(env_name, env_prefix),
            ],
        )

    # ---- Clone and install METEOR (tag v1.6.0) -------------------------------
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    meteor_dest = src_dir / "METEOR"
    print("\nSetting up METEOR (tag v1.6.0)…")
    try:
        cloned, used = clone_or_update_repo(
            meteor_dest,
            ssh_url="git@github.com:benmsanderson/METEOR.git",
            https_url="https://github.com/benmsanderson/METEOR.git",
            ref="v1.6.0",
            is_tag=True,
        )
        if not cloned:
            print("WARNING: METEOR was not cloned/updated successfully.", file=sys.stderr)
            return 5
        print(f"METEOR available at {meteor_dest} (via {used}).")
    except Exception as exc:
        print(f"ERROR: Failed to obtain METEOR: {exc}", file=sys.stderr)
        return 6

    _install_repo(conda_cmd, env_name, env_prefix, meteor_dest, label="METEOR")

    # ---- Install local FYS9429 package from pyproject.toml ------------------
    print("\nInstalling local FYS9429 package (editable from repo root)…")
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        print(
            f"ERROR: pyproject.toml not found at {pyproject_path}",
            file=sys.stderr,
        )
        return 11
    try:
        _run_in_env(
            conda_cmd,
            env_name,
            env_prefix,
            ["python", "-m", "pip", "install", "-e", "."],
            cwd=project_root,
        )
        print("Successfully installed local FYS9429 package (editable).")
    except subprocess.CalledProcessError:
        print(
            "ERROR: editable install from repo root failed. "
            "Check pyproject.toml and package discovery settings.",
            file=sys.stderr,
        )
        return 12

    # ---- Clone and install general_backend -----------------------------------
    backend_dest = src_dir / "general_backend"
    print("\nSetting up general_backend…")
    try:
        cloned, used = clone_or_update_repo(
            backend_dest,
            ssh_url="git@github.com:Johannesfjeldsaa/general_backend.git",
            https_url="https://github.com/Johannesfjeldsaa/general_backend.git",
            ref=args.backend_branch,
            is_tag=False,
        )
        if not cloned:
            print("WARNING: general_backend was not cloned/updated successfully.", file=sys.stderr)
            return 5
        print(f"general_backend available at {backend_dest} (via {used}).")
    except Exception as exc:
        print(f"ERROR: Failed to obtain general_backend: {exc}", file=sys.stderr)
        return 6

    # Try with [dev] extras first, fall back to plain install.
    success = _install_repo(
        conda_cmd, env_name, env_prefix, backend_dest,
        label="general_backend", extra="dev",
    )
    if not success:
        _install_repo(
            conda_cmd, env_name, env_prefix, backend_dest,
            label="general_backend (without extras)",
        )

    # ---- Resolve env python --------------------------------------------------
    env_prefix_resolved = _get_env_prefix(conda_cmd, env_name, env_prefix)
    env_python = env_prefix_resolved / "bin" / "python"
    if not env_python.exists():
        print(f"ERROR: python not found at {env_python}", file=sys.stderr)
        return 9

    # ---- ESMF compatibility shim ---------------------------------------------
    print("\nEnsuring ESMF / esmpy compatibility shim for xesmf…")
    _ensure_esmf_module_alias(conda_cmd, env_name, env_prefix)

    # ---- Optional: Jupyter kernel registration --------------------------------
    if args.register_kernel:
        print(
            f"\nRegistering Jupyter kernel '{args.kernel_name}' "
            f"({args.kernel_display})…"
        )
        try:
            _run_in_env(
                conda_cmd, env_name, env_prefix,
                ["python", "-m", "pip", "install", "--upgrade", "ipykernel"],
            )
            _run_in_env(
                conda_cmd, env_name, env_prefix,
                [
                    "python", "-m", "ipykernel", "install",
                    "--user",
                    "--name", args.kernel_name,
                    "--display-name", args.kernel_display,
                ],
            )
            print("Kernel registered.")
        except subprocess.CalledProcessError:
            print(
                "WARNING: failed to register ipykernel; setup will continue.\n"
                "Try running without --register-kernel, then register manually later.",
                file=sys.stderr,
            )

    # ---- VS Code settings ----------------------------------------------------
    if not args.no_vscode:
        vscode_dir = project_root / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        site_packages = _get_site_packages(conda_cmd, env_name, env_prefix)
        settings = {
            "python.defaultInterpreterPath": str(env_python),
            "python.terminal.activateEnvironment": True,
            "python.analysis.extraPaths": [
                site_packages,
                "${workspaceFolder}/src",
            ],
            "python.autoComplete.extraPaths": [
                site_packages,
                "${workspaceFolder}/src",
            ],
            "terminal.integrated.env.linux": {
                "PYTHONPATH": f"{site_packages}:${{workspaceFolder}}/src"
            },
        }
        if use_system_modules:
            # A login shell sources /etc/profile.d/lmod.sh so that
            # 'module load' commands in .envrc work in the VS Code terminal.
            settings["terminal.integrated.defaultProfile.linux"] = "bash-login"
            settings["terminal.integrated.profiles.linux"] = {
                "bash-login": {"path": "bash", "args": ["-l"]}
            }
        settings_path = vscode_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))
        print(f"\nWrote VS Code settings to {settings_path}")

    # ---- .envrc for direnv ---------------------------------------------------
    activation_target = str(env_prefix_resolved) if env_prefix else env_name

    module_block = ""
    if use_system_modules:
        loads = "\n".join(f"    module load {m}" for m in _SYSTEM_MODULES)
        module_block = (
            "\n"
            "# Load HPC system modules for GPU / PyTorch support.\n"
            "if [ -n \"$LMOD_CMD\" ] || command -v module &> /dev/null; then\n"
            f"{loads}\n"
            "fi\n"
        )

    envrc_text = (
        "# Auto-activate the FYS9429 conda environment (requires direnv).\n"
        "\n"
        "if command -v conda &> /dev/null; then\n"
        "    eval \"$(conda shell.bash hook)\"\n"
        f"    conda activate {activation_target}\n"
        "fi\n"
        + module_block
    )
    envrc_path = project_root / ".envrc"
    envrc_path.write_text(envrc_text)
    print(f"Wrote .envrc to {envrc_path}")

    # ---- Verify imports -------------------------------------------------------
    print("\nVerifying core imports (conda env)…")
    check_code = (
        "import xarray, numpy, pandas, xesmf, statsmodels, sklearn; "
        "import sys; "
        "print(f'Python  {sys.version.split()[0]}'); "
        "print(f'xarray  {xarray.__version__}'); "
        "print(f'numpy   {numpy.__version__}'); "
        "print(f'xesmf   {xesmf.__version__}'); "
        "print('Core imports OK')"
    )
    try:
        _run_in_env(conda_cmd, env_name, env_prefix, ["python", "-c", check_code])
    except subprocess.CalledProcessError:
        print(
            "WARNING: core import verification failed — check the output above.",
            file=sys.stderr,
        )

    if use_system_modules and lmod_init:
        print("Verifying PyTorch import via system modules…")
        torch_check = (
            "import torch; "
            "print(f'torch   {torch.__version__} "
            "(CUDA available: {torch.cuda.is_available()})')"
        )
        try:
            _run_in_env_with_modules(
                conda_cmd, env_name, env_prefix,
                ["python", "-c", torch_check],
                _SYSTEM_MODULES,
                lmod_init,
            )
        except subprocess.CalledProcessError:
            print(
                "WARNING: PyTorch import via system modules failed.\n"
                "You may need to run 'module load' manually before using torch.",
                file=sys.stderr,
            )

    # ---- Done ----------------------------------------------------------------
    if env_prefix is not None:
        activate_hint = f"conda activate {env_prefix_resolved}"
    else:
        activate_hint = f"conda activate {env_name}"

    print(f"\nActivate with:  {activate_hint}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
