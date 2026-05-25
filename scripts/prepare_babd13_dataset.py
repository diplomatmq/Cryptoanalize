from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

LABEL_MAP = {
    "3": "exchange",
    "10": "miner",
    "11": "mixer",
    "12": "individual",
}

DROP_COLUMNS = {"account", "SW", "label"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="BABD-13.csv", help="Path to BABD-13.csv")
    parser.add_argument("--output", default="data/babd13_wallet4.csv", help="Output CSV")
    parser.add_argument("--per-class", type=int, default=5000, help="Max rows per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        feature_fields = [
            name for name in reader.fieldnames if name not in DROP_COLUMNS
        ]
        output_fields = feature_fields + ["label"]

        buffers: dict[str, list[dict[str, str]]] = {label: [] for label in LABEL_MAP.values()}
        seen_counts: dict[str, int] = {label: 0 for label in LABEL_MAP.values()}

        for row in reader:
            raw_label = row.get("label")
            mapped = LABEL_MAP.get(raw_label)
            if mapped is None:
                continue

            seen_counts[mapped] += 1
            clean_row = {field: row.get(field, "") for field in feature_fields}
            clean_row["label"] = mapped

            bucket = buffers[mapped]
            if len(bucket) < args.per_class:
                bucket.append(clean_row)
                continue

            # Reservoir sampling to keep a uniform sample per class.
            idx = random.randint(0, seen_counts[mapped] - 1)
            if idx < args.per_class:
                bucket[idx] = clean_row

    total_rows = sum(len(rows) for rows in buffers.values())
    if total_rows == 0:
        print("No rows matched the selected labels.")
        return

    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=output_fields)
        writer.writeheader()
        for label in sorted(buffers.keys()):
            for row in buffers[label]:
                writer.writerow(row)

    print("Prepared dataset:")
    for label, rows in buffers.items():
        print(f"  {label}: {len(rows)} rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
