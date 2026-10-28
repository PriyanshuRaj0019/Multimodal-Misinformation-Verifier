
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
FEVER_DIR = ROOT / "data" / "raw" / "fever"
WIKI_DIR = FEVER_DIR / "wiki-pages.zip" / "wiki-pages"

DATASETS = [
    FEVER_DIR / "train.jsonl",
    FEVER_DIR / "shared_task_dev.jsonl",
]


def load_references():
    references = defaultdict(set)

    for dataset_path in DATASETS:
        with dataset_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)

                for group in row.get("evidence") or []:
                    if not isinstance(group, list):
                        continue

                    for item in group:
                        if not isinstance(item, list) or len(item) < 4:
                            continue

                        page = item[2]
                        sentence_id = item[3]

                        if not isinstance(page, str) or not page:
                            continue

                        if sentence_id is None:
                            continue

                        try:
                            sentence_id = int(sentence_id)
                        except (TypeError, ValueError):
                            continue

                        references[page].add(sentence_id)

    return references


def normalized(value, form):
    return unicodedata.normalize(form, value)


def main():
    print("Loading FEVER evidence references...")
    references = load_references()
    required_pages = set(references)

    corpus_files = sorted(WIKI_DIR.glob("wiki-*.jsonl"))
    if not corpus_files:
        raise FileNotFoundError(f"No corpus files found: {WIKI_DIR}")

    print(f"Required pages: {len(required_pages):,}")
    print(f"Corpus files: {len(corpus_files):,}")
    print("Scanning corpus; this may take several minutes...")

    exact_pages = set()
    nfc_pages = set()
    nfd_pages = set()
    sentence_ids_by_page = {}

    required_nfc = {
        normalized(page, "NFC"): page
        for page in required_pages
    }
    required_nfd = {
        normalized(page, "NFD"): page
        for page in required_pages
    }

    for file_index, path in enumerate(corpus_files, start=1):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                page = record.get("id")

                if not isinstance(page, str) or not page:
                    continue

                targets = set()

                if page in required_pages:
                    exact_pages.add(page)
                    targets.add(page)

                nfc_target = required_nfc.get(normalized(page, "NFC"))
                if nfc_target is not None:
                    nfc_pages.add(nfc_target)
                    targets.add(nfc_target)

                nfd_target = required_nfd.get(normalized(page, "NFD"))
                if nfd_target is not None:
                    nfd_pages.add(nfd_target)
                    targets.add(nfd_target)

                if not targets:
                    continue

                lines = record.get("lines", "")
                available_ids = set()

                if isinstance(lines, str):
                    for sentence_line in lines.splitlines():
                        sentence_id_text, separator, _ = (
                            sentence_line.partition("\t")
                        )

                        if not separator:
                            continue

                        try:
                            available_ids.add(int(sentence_id_text))
                        except ValueError:
                            continue

                for target in targets:
                    sentence_ids_by_page[target] = available_ids

        if file_index % 10 == 0 or file_index == len(corpus_files):
            print(f"Scanned {file_index}/{len(corpus_files)} files")

    missing_pages = required_pages - exact_pages
    nfc_only = (nfc_pages - exact_pages) & missing_pages
    nfd_only = (nfd_pages - exact_pages) & missing_pages

    missing_sentences = []
    for page, required_ids in references.items():
        if page not in exact_pages:
            continue

        available_ids = sentence_ids_by_page.get(page, set())

        for sentence_id in sorted(required_ids - available_ids):
            missing_sentences.append((page, sentence_id))

    print("\n========== DIAGNOSTIC SUMMARY ==========")
    print(f"Exact page matches: {len(exact_pages):,}/{len(required_pages):,}")
    print(f"Missing exact page titles: {len(missing_pages):,}")
    print(f"Missing pages matched via NFC normalization: {len(nfc_only):,}")
    print(f"Missing pages matched via NFD normalization: {len(nfd_only):,}")
    print(
        "Missing sentence references on exact-matched pages:",
        len(missing_sentences),
    )

    if missing_pages:
        print("\nSample missing page titles:")
        for page in sorted(missing_pages)[:15]:
            print(" -", repr(page))

    if nfc_only:
        print("\nSample NFC normalization matches:")
        for page in sorted(nfc_only)[:10]:
            print(" -", repr(page), "=>", repr(normalized(page, "NFC")))

    if nfd_only:
        print("\nSample NFD normalization matches:")
        for page in sorted(nfd_only)[:10]:
            print(" -", repr(page), "=>", repr(normalized(page, "NFD")))

    if missing_sentences:
        print("\nSample missing sentence IDs:")
        for page, sentence_id in missing_sentences[:15]:
            print(f" - {page!r} | sentence {sentence_id}")


if __name__ == "__main__":
    main()
