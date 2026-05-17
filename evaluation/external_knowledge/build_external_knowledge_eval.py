import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent

SOURCE_FILES = [
    "mavlink_ardupilotmega_eval.jsonl",
    "ardupilot_logs_eval.jsonl",
    "mavlink_common_eval.jsonl",
]

OUTPUT_FILE = "external_knowledge_eval.jsonl"

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse {path.name} line {line_no}: {e}"
                ) from e
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    output_path = EVAL_DIR / OUTPUT_FILE

    if output_path.exists():
        output_path.unlink()
        print(f"Removed existing {OUTPUT_FILE}")

    combined: list[dict] = []
    seen_ids: set[str] = set()

    for filename in SOURCE_FILES:
        source_path = EVAL_DIR / filename
        if not source_path.exists():
            print(f"WARNING: {filename} not found, skipping.")
            continue

        records = load_jsonl(source_path)
        print(f"Loaded {len(records):3d} records from {filename}")

        for record in records:
            record_id = record.get("id")
            if record_id and record_id in seen_ids:
                print(f"  WARNING: duplicate id {record_id} — skipping")
                continue
            if record_id:
                seen_ids.add(record_id)
            combined.append(record)

    write_jsonl(output_path, combined)
    print(f"\nWrote {len(combined)} records to {OUTPUT_FILE}")

    categories: dict[str, int] = {}
    for r in combined:
        cat = r.get("category", "uncategorized")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:35s} {count:4d}")


if __name__ == "__main__":
    main()
