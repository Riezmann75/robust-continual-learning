"""Shared implementations for robust continual-learning experiments."""
import hashlib
from pathlib import Path
import shutil


def library_source_hashes():
    """Include executable shared code in new run provenance."""
    return {f'libs/{path.name}': hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob('*.py'))}


def snapshot_source(runner_directory, snapshot, name):
    """Copy runner or library code into a self-contained source snapshot."""
    source = (Path(__file__).parent / Path(name).name
              if name.startswith('libs/') else runner_directory / name)
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
