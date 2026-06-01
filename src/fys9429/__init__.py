from pathlib import Path
from fys9429 import ae_diagnostics  # noqa: F401
from fys9429 import ae_experiment_runner  # noqa: F401


def get_src_dir() -> Path:
    """Get the source directory of the package."""
    return Path(__file__).parent

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return get_src_dir().parent

SRC_DIR = get_src_dir()
PROJECT_ROOT = get_project_root()