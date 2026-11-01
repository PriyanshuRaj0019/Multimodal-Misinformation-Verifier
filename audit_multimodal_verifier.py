"""
Read-only project audit for Multimodal Misinformation Verifier.

Run from the project root using the existing .venv.

Does NOT:
- Modify source code or datasets
- Install dependencies
- Rebuild the SQLite evidence index
- Train models
- Commit or push to Git

Writes one audit report under reports/audit/.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path.cwd()
REPORT_DIR = ROOT / "reports" / "audit"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
REPORT_PATH = REPORT_DIR / f"project_audit_{TIMESTAMP}.txt"

MAX_FILES = 3000
MAX_TEXT_BYTES = 250_000

SCAN_ROOTS = [
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
    "reports",
    "data/raw/fever",
    "data/processed",
]

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "datasets": "datasets",
    "evaluate": "evaluate",
    "rank-bm25": "rank_bm25",
    "faiss": "faiss",
    "pytest": "pytest",
    "ruff": "ruff",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "Pillow": "PIL",
    "pytesseract": "pytesseract",
    "opencv-python": "cv2",
    "faster-whisper": "faster_whisper",
    "rapidfuzz": "rapidfuzz",
    "joblib": "joblib",
}

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg",
}


def heading(title: str) -> str:
    return f"\n{'=' * 88}\n{title}\n{'=' * 88}\n"


def human_size(size: int) -> str:
    value = float(size)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{size} B"


def run_command(args: list[str], timeout: int = 30) -> str:
    """Run a command without invoking a shell or modifying project files."""
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )

        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()

        if error:
            output += ("\n" if output else "") + "[stderr]\n" + error

        return (
            f"Exit code: {result.returncode}\n"
            f"{output or '(no output)'}"
        )

    except FileNotFoundError:
        return f"Command not available: {args[0]}"

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds: {args}"

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def collect_files(roots: list[Path]) -> list[Path]:
    """Collect files while skipping virtual environments and caches."""
    files = []

    excluded_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
    }

    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            files.append(root)
            continue

        for current, dirs, names in os.walk(root):
            dirs[:] = [
                name for name in dirs
                if name not in excluded_dirs
            ]

            for name in names:
                files.append(Path(current) / name)

                if len(files) >= MAX_FILES:
                    return sorted(files, key=lambda p: str(p).lower())

    return sorted(set(files), key=lambda p: str(p).lower())


def git_audit() -> str:
    commands = {
        "Current branch": ["git", "branch", "--show-current"],
        "Working tree status": ["git", "status", "--short", "--branch"],
        "Recent commits": [
            "git", "log", "-5", "--oneline", "--decorate"
        ],
        "Unstaged diff summary": ["git", "diff", "--stat"],
        "Staged diff summary": ["git", "diff", "--cached", "--stat"],
    }

    output = []

    for title, command in commands.items():
        output.append(f"\n--- {title} ---")
        output.append(run_command(command))

    return "\n".join(output)


def environment_audit() -> str:
    output = [
        f"Project root: {ROOT}",
        f"Python executable: {sys.executable}",
        f"Python version: {sys.version}",
        f"Platform: {platform.platform()}",
        f"sys.prefix: {sys.prefix}",
        f"sys.base_prefix: {sys.base_prefix}",
        (
            "Virtual environment active: "
            f"{sys.prefix != sys.base_prefix}"
        ),
        "",
        "Package availability:",
    ]

    for package, module in PACKAGES.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False

        output.append(
            f"  {package}: "
            f"{'AVAILABLE' if available else 'NOT FOUND'}"
        )

    output.extend([
        "",
        "--- pip version ---",
        run_command([sys.executable, "-m", "pip", "--version"]),
    ])

    return "\n".join(output)


def gpu_audit() -> str:
    return (
        "--- nvidia-smi ---\n"
        + run_command(["nvidia-smi"], timeout=15)
        + "\n\n--- CUDA compiler ---\n"
        + run_command(["nvcc", "--version"], timeout=10)
    )


def inventory_audit() -> str:
    roots = [
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "configs",
    ROOT / "docs",
    ROOT / "reports",
]
    roots.extend([
        ROOT / "README.md",
        ROOT / ".gitignore",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "environment.yml",
        ROOT / "setup.py",
        ROOT / "setup.cfg",
        ROOT / "pytest.ini",
    ])

    files = collect_files(roots)
    output = [
        f"Files found: {len(files)}",
        f"Listing cap: {MAX_FILES}",
        "",
    ]

    for path in files:
        try:
            relative = path.relative_to(ROOT)
            output.append(
                f"{human_size(path.stat().st_size):>12}  {relative}"
            )
        except OSError as exc:
            output.append(f"Could not inspect {path}: {exc}")

    if len(files) >= MAX_FILES:
        output.append("\nWARNING: File listing reached the configured cap.")

    return "\n".join(output)


def index_audit() -> str:
    """Inspect the existing SQLite index in read-only mode."""
    index = ROOT / "data" / "processed" / "wiki_sentence_index.sqlite3"

    if not index.exists():
        return f"Evidence index NOT FOUND: {index}"

    output = [
        f"Index path: {index}",
        f"Index size: {human_size(index.stat().st_size)}",
        f"Last modified: {time.ctime(index.stat().st_mtime)}",
    ]

    try:
        uri = index.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            objects = connection.execute(
                """
                SELECT type, name, sql
                FROM sqlite_master
                WHERE type IN ('table', 'view', 'index')
                ORDER BY type, name
                """
            ).fetchall()

            output.append("\nSQLite schema objects:")

            for object_type, name, sql in objects[:100]:
                output.append(
                    f"\n[{object_type}] {name}\n{sql or '(no SQL definition)'}"
                )

            try:
                count = connection.execute(
                    "SELECT COUNT(*) FROM sentences"
                ).fetchone()[0]

                output.append(f"\nSentence row count: {count}")

            except sqlite3.Error as exc:
                output.append(f"\nSentence count unavailable: {exc}")


    except Exception as exc:
        output.append(
            f"Read-only index inspection failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return "\n".join(output)


def data_and_report_audit() -> str:
    targets = [
        ROOT / "data" / "raw" / "fever",
        ROOT / "data" / "processed",
        ROOT / "reports" / "retrieval",
        ROOT / "reports",
    ]

    output = []

    for target in targets:
        output.append(f"\n--- {target.relative_to(ROOT)} ---")

        if not target.exists():
            output.append("NOT FOUND")
            continue

        if target.is_file():
            output.append(
                f"{target.name}: {human_size(target.stat().st_size)}"
            )
            continue

        children = sorted(target.iterdir(), key=lambda p: p.name.lower())

        if not children:
            output.append("(empty)")
            continue

        for item in children[:200]:
            try:
                if item.is_file():
                    output.append(
                        f"{human_size(item.stat().st_size):>12}  {item.name}"
                    )
                else:
                    output.append(f"{'[DIR]':>12}  {item.name}/")
            except OSError as exc:
                output.append(f"Could not inspect {item.name}: {exc}")

        if len(children) > 200:
            output.append(
                f"... {len(children) - 200} more entries omitted"
            )

    return "\n".join(output)


def source_outline_audit() -> str:
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "configs",
    ]

    files = collect_files(roots)
    output = []

    for path in files:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            size = path.stat().st_size

            output.append(
                f"\n--- {path.relative_to(ROOT)} ({human_size(size)}) ---"
            )

            if size > MAX_TEXT_BYTES:
                output.append("(Skipped: file exceeds preview size limit)")
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            if path.suffix.lower() == ".py":
                declarations = []

                for line_number, line in enumerate(
                    text.splitlines(), start=1
                ):
                    stripped = line.strip()

                    if stripped.startswith((
                        "import ",
                        "from ",
                        "class ",
                        "def ",
                        "async def ",
                    )):
                        declarations.append(
                            f"{line_number}: {line[:220]}"
                        )

                output.extend(
                    declarations[:250]
                    if declarations
                    else ["No import/class/function declarations found."]
                )

            else:
                lines = text.splitlines()
                output.extend(lines[:100])

                if len(lines) > 100:
                    output.append("... preview truncated")

        except Exception as exc:
            output.append(
                f"Read error: {type(exc).__name__}: {exc}"
            )

    return "\n".join(output) or "No source/config files found."


def keyword_scan_audit() -> str:
    patterns = [
        "train_test_split",
        "GroupShuffleSplit",
        "Stratified",
        "LogisticRegression",
        "TfidfVectorizer",
        "BM25",
        "Recall@",
        "recall_at",
        "macro_f1",
        "classification_report",
        "confusion_matrix",
        "calibration",
        "NOT ENOUGH INFO",
        "evidence",
        "page_title",
        "sentence_id",
        "pytest",
        "TODO",
        "FIXME",
    ]

    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "reports",
    ]

    files = collect_files(roots)
    searchable = []

    for path in files:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue

            searchable.append((
                path,
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
            ))

        except OSError:
            continue

    output = []

    for pattern in patterns:
        output.append(f"\n[{pattern}]")
        hits = []

        for path, lines in searchable:
            for number, line in enumerate(lines, start=1):
                if pattern.casefold() in line.casefold():
                    hits.append(
                        f"{path.relative_to(ROOT)}:{number}: "
                        f"{line.strip()[:180]}"
                    )

                    if len(hits) >= 8:
                        break

            if len(hits) >= 8:
                break

        output.extend(hits or ["No matches found."])

    return "\n".join(output)


def key_document_audit() -> str:
    documents = [
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "data-card.md",
        ROOT / "docs" / "model-card.md",
        ROOT / "reports" / "retrieval" / "baseline_verifier.md",
        ROOT / "reports" / "retrieval" / "baseline_verifier_metrics.json",
    ]

    output = []

    for path in documents:
        output.append(f"\n--- {path.relative_to(ROOT)} ---")

        if not path.exists():
            output.append("NOT FOUND")
            continue

        size = path.stat().st_size
        output.append(f"Size: {human_size(size)}")

        if size == 0:
            output.append("(Empty file)")
            continue

        if size > MAX_TEXT_BYTES:
            output.append("(Skipped: file exceeds preview size limit)")
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            output.extend(lines[:180])

            if len(lines) > 180:
                output.append("... preview truncated")

        except Exception as exc:
            output.append(f"Read error: {exc}")

    return "\n".join(output)


def main() -> int:
    if not (ROOT / ".git").exists():
        print(
            "ERROR: Run this script from the project root "
            "(the folder containing .git)."
        )
        print(f"Current directory: {ROOT}")
        return 2

    # This is the only directory/file this script creates.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sections = [
        ("PROJECT PATH", str(ROOT)),
        ("GIT STATE", git_audit()),
        ("PYTHON AND DEPENDENCIES", environment_audit()),
        ("GPU AND CUDA", gpu_audit()),
        ("FILE INVENTORY", inventory_audit()),
        ("DATA AND EXISTING REPORTS", data_and_report_audit()),
        ("SQLITE EVIDENCE INDEX", index_audit()),
        ("SOURCE OUTLINE AND CONFIGURATION", source_outline_audit()),
        ("PHASE 5/6/7 KEYWORD SCAN", keyword_scan_audit()),
        ("KEY DOCUMENTS AND BASELINE METRICS", key_document_audit()),
    ]

    report = "\n".join(
        heading(title) + body
        for title, body in sections
    )

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nAUDIT COMPLETE")
    print(f"Project root : {ROOT}")
    print(f"Report path  : {REPORT_PATH}")
    print(f"Report size  : {human_size(REPORT_PATH.stat().st_size)}")
    print("\nNo packages were installed.")
    print("No source files or datasets were modified.")
    print("The SQLite evidence index was opened in read-only mode.")
    print("\nPrint the report with:")
    print(f'Get-Content "{REPORT_PATH}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

