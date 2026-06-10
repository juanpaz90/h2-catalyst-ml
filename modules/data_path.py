# from pathlib import Path

# def find_project_root(start: Path) -> Path:
#     for p in [start, *start.parents]:
#         if (p / 'modules').is_dir():
#             return p
#     raise FileNotFoundError("Could not find project root containing 'modules/'")

# def resolve_dataset_path(relative_path: str, start: Path | None = None) -> str:
#     """
#     Resolve a dataset path relative to the shared data folder next to the project root.
#     """
#     start = (start or Path.cwd()).resolve()
#     project_root = find_project_root(start)
#     candidate = project_root.parent / relative_path
#     if not candidate.exists():
#         raise FileNotFoundError(f'Database not found at: {candidate}')
#     return str(candidate.resolve())

from pathlib import Path
from typing import Iterable

def find_project_root(start: Path, indicator_dirs: str | Iterable[str] = ('modules',)) -> Path:
    """
    Traverse up the directory tree to find a project root containing specific directories.
    """
    # Normalize to a tuple if a single string is passed
    if isinstance(indicator_dirs, str):
        indicator_dirs = (indicator_dirs,)
        
    for p in [start, *start.parents]:
        # Return the path if ANY of the indicator directories exist here
        if any((p / indicator).is_dir() for indicator in indicator_dirs):
            return p
            
    raise FileNotFoundError(f"Could not find project root containing any of: {indicator_dirs}")


def resolve_dataset_path(relative_path: str, start: Path | None = None, indicator_dirs: str | Iterable[str] = ('modules',)) -> str:
    """
    Resolve a dataset path relative to the shared data folder next to the project root.
    """
    start = (start or Path.cwd()).resolve()
    
    # Pass the indicator directories down to the root finder
    project_root = find_project_root(start, indicator_dirs)
    
    candidate = project_root.parent / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f'Database not found at: {candidate}')
        
    return str(candidate.resolve())