from pathlib import Path
from uuid import uuid4
import sys

from datasets import load_dataset

ROOT = Path("data")
FILES_PER_FOLDER = 1000

DATASETS = [
    ("pretrain", "gszauer/Gab100MPretrain"),
    ("finetune", "gszauer/Gab100MFinetune"),
]


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def output_path(root, index):
    folder = root / f"{index // FILES_PER_FOLDER:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{uuid4().hex}.txt"


def download_dataset(name, repo):
    folder = ROOT / name
    written = 0
    skipped = 0

    try:
        folder.mkdir(parents=True, exist_ok=True)

        if next(folder.rglob("*.txt"), None) is not None:
            fail(f"{folder} already contains .txt files. Delete it before running again.")

        rows = load_dataset(repo, split="train", streaming=True)

        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or len(text) == 0:
                skipped += 1
                continue

            output_path(folder, written).write_text(text, encoding="utf-8")
            written += 1

            if written % FILES_PER_FOLDER == 0:
                print(f"[{name}] wrote {written:,} files")

    except KeyboardInterrupt:
        fail(f"Stopped while downloading {name}. Wrote {written:,} files.")
    except OSError as error:
        fail(f"Could not write files for {name}. Check disk space and permissions.\n{error}")
    except Exception as error:
        fail(f"Could not download {name} from {repo}.\n{error}")

    print(f"[{name}] finished. Wrote {written:,} files. Skipped {skipped:,} rows.")
    return written


def main():
    total = 0

    for name, repo in DATASETS:
        total += download_dataset(name, repo)

    print(f"Finished all datasets. Wrote {total:,} files.")


if __name__ == "__main__":
    main()
