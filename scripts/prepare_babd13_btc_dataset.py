from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from btc_wallet_features import extract_btc_features, fetch_mempool_transactions

LABEL_MAP = {
    "3": "exchange",
    "10": "miner",
    "11": "mixer",
    "12": "individual",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="BABD-13.csv", help="Path to BABD-13.csv")
    parser.add_argument("--out", default="data/babd13_btc_features.csv", help="Output dataset CSV")
    parser.add_argument("--per-class", type=int, default=200, help="Addresses per class")
    parser.add_argument("--max-tx", type=int, default=300, help="Max tx per address")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=25, help="Request timeout seconds")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests")
    parser.add_argument("--retries", type=int, default=2, help="Request retry attempts")
    parser.add_argument("--backoff", type=float, default=1.0, help="Retry backoff seconds")
    parser.add_argument("--max-errors", type=int, default=50, help="Stop after N errors")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[str]] = {label: [] for label in LABEL_MAP.values()}
    seen_counts: dict[str, int] = {label: 0 for label in LABEL_MAP.values()}

    with input_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        for row in reader:
            raw_label = row.get("label")
            mapped = LABEL_MAP.get(raw_label)
            if mapped is None:
                continue

            address = str(row.get("account", "")).strip()
            if not address:
                continue

            seen_counts[mapped] += 1
            bucket = buffers[mapped]
            if len(bucket) < args.per_class:
                bucket.append(address)
                continue

            idx = random.randint(0, seen_counts[mapped] - 1)
            if idx < args.per_class:
                bucket[idx] = address

    row_count = 0
    error_count = 0
    stop = False

    with out_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer: csv.DictWriter | None = None
        for label, addresses in buffers.items():
            if stop:
                break
            for address in addresses:
                if stop:
                    break
                try:
                    transactions = fetch_mempool_transactions(
                        address,
                        max_tx=args.max_tx,
                        request_timeout=args.timeout,
                        sleep_seconds=args.sleep,
                        max_retries=args.retries,
                        backoff_seconds=args.backoff,
                    )
                except Exception as exc:
                    error_count += 1
                    print(f"[WARN] {address}: {exc}")
                    if error_count >= args.max_errors:
                        print("Reached max errors. Stopping early.")
                        stop = True
                    continue

                if not transactions:
                    print(f"[WARN] {address}: no transactions")
                    continue

                features = extract_btc_features(transactions, address)
                features["label"] = label
                row = {k: str(v) for k, v in features.items()}

                if writer is None:
                    fieldnames = sorted(row.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(row)
                row_count += 1

    if row_count == 0:
        print("No rows collected. Check API access and address sampling.")
        return

    print(f"Dataset saved to: {out_path}")
    print(f"Rows: {row_count}")


if __name__ == "__main__":
    main()
