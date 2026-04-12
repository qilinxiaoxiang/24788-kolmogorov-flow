"""Environment detection and path resolution.

Works in three contexts:
  1. Local dev (macOS, project synced to Google Drive)
  2. Colab with Drive mounted
  3. Any other environment with the repo checked out

Usage:
    from src.env import PROJECT_ROOT, DATA_DIR, CKPT_DIR, RESULTS_DIR, IN_COLAB
"""
from pathlib import Path
import os
import sys


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = _detect_colab()


def _find_project_root() -> Path:
    """Walk upward from this file looking for a marker (requirements.txt)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    # Fallback: parent of src/
    return here.parent.parent


if IN_COLAB:
    # Mount Drive if not already mounted
    if not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")

    # The user syncs via Drive for Desktop, so the repo may live under:
    #   /content/drive/MyDrive/...                         (if in My Drive)
    #   /content/drive/Othercomputers/My MacBook Pro/...   (if on a synced Mac)
    # We search those roots for a folder containing src/env.py.
    _marker = "src/env.py"
    _search_roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Othercomputers"),
    ]
    # Computer names under Othercomputers to probe (Drive for Desktop syncs)
    _computer_suffixes = [
        "My MacBook Pro",
        "MacBook Pro",
        "My MacBook Air",
        "MacBook Air",
        "MyMacBook",
    ]

    def _find_repo() -> Path | None:
        import subprocess
        # Fast path: common manual placements
        for root in _search_roots:
            if not root.exists():
                continue
            fast_paths = [
                root / "courses/24788-Intro_of_DL/project/code",
                root / "24788-Intro_of_DL/project/code",
                root / "project/code",
            ]
            # Drive for Desktop adds a <Computer Name> layer under Othercomputers
            if root.name == "Othercomputers":
                for comp in _computer_suffixes:
                    fast_paths.extend([
                        root / comp / "courses/24788-Intro_of_DL/project/code",
                        root / comp / "Desktop/courses/24788-Intro_of_DL/project/code",
                    ])
            for p in fast_paths:
                if (p / _marker).exists():
                    return p
        # Slow path: `find` for the marker (Drive API is slow, but this is O(N))
        for root in _search_roots:
            if not root.exists():
                continue
            try:
                out = subprocess.check_output(
                    ["find", str(root), "-maxdepth", "8", "-name", "env.py",
                     "-path", "*/src/env.py", "-type", "f"],
                    stderr=subprocess.DEVNULL, timeout=60,
                ).decode().strip().splitlines()
                for hit in out:
                    p = Path(hit).parent.parent
                    if (p / "requirements.txt").exists():
                        return p
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue
        return None

    found = _find_repo()
    if found is None:
        raise RuntimeError(
            "Could not locate project repo in Drive. Searched "
            f"{_search_roots} for src/env.py. "
            "If your sync path differs, set PROJECT_ROOT manually."
        )
    PROJECT_ROOT = found
else:
    PROJECT_ROOT = _find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = Path(os.environ.get("CKPT_DIR_OVERRIDE", str(PROJECT_ROOT / "checkpoints")))
RESULTS_DIR = PROJECT_ROOT / "results"

for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Make `from src.xxx import yyy` work when running notebooks
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summary() -> str:
    return (
        f"IN_COLAB     = {IN_COLAB}\n"
        f"PROJECT_ROOT = {PROJECT_ROOT}\n"
        f"DATA_DIR     = {DATA_DIR}\n"
        f"CKPT_DIR     = {CKPT_DIR}\n"
        f"RESULTS_DIR  = {RESULTS_DIR}"
    )


if __name__ == "__main__":
    print(summary())
