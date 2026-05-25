from __future__ import annotations

from typing import Any
import math
import time

import numpy as np
import requests

DEFAULT_BASE_URL = "https://mempool.space/api"


def fetch_mempool_transactions(
    address: str,
    max_tx: int = 300,
    request_timeout: int = 25,
    base_url: str = DEFAULT_BASE_URL,
    sleep_seconds: float = 0.0,
    max_retries: int = 2,
    backoff_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    next_txid: str | None = None

    while len(transactions) < max_tx:
        if next_txid:
            url = f"{base_url}/address/{address}/txs/chain"
            params = {"after_txid": next_txid}
        else:
            url = f"{base_url}/address/{address}/txs"
            params = None

        batch = _request_json(
            url,
            params=params,
            timeout=request_timeout,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        if not isinstance(batch, list) or not batch:
            break

        transactions.extend(batch)
        next_txid = str(batch[-1].get("txid", "")) or None
        if not next_txid:
            break
        if len(batch) < 25:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return transactions[:max_tx]


def _request_json(
    url: str,
    params: dict[str, Any] | None,
    timeout: int,
    max_retries: int,
    backoff_seconds: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(backoff_seconds * (attempt + 1))
    if last_exc:
        raise last_exc
    return []


def extract_btc_features(
    transactions: list[dict[str, Any]],
    wallet_address: str,
) -> dict[str, float]:
    wallet = wallet_address.strip().lower()

    in_values: list[float] = []
    out_values: list[float] = []
    in_count = 0
    out_count = 0

    in_sources: dict[str, int] = {}
    out_targets: dict[str, int] = {}

    coinbase_in_count = 0

    for tx in transactions:
        vin_items = tx.get("vin") if isinstance(tx.get("vin"), list) else []
        vout_items = tx.get("vout") if isinstance(tx.get("vout"), list) else []

        vin_addrs: list[str] = []
        vout_addrs: list[str] = []
        is_coinbase = False

        out_value = 0.0
        for vin in vin_items:
            if isinstance(vin, dict) and vin.get("is_coinbase"):
                is_coinbase = True
            prevout = vin.get("prevout") if isinstance(vin, dict) else None
            if isinstance(prevout, dict):
                addr = _normalize_address(prevout.get("scriptpubkey_address"))
                if addr:
                    vin_addrs.append(addr)
                    if addr == wallet:
                        out_value += float(_safe_int(prevout.get("value")))

        in_value = 0.0
        for vout in vout_items:
            if not isinstance(vout, dict):
                continue
            addr = _normalize_address(vout.get("scriptpubkey_address"))
            if addr:
                vout_addrs.append(addr)
                if addr == wallet:
                    in_value += float(_safe_int(vout.get("value")))

        wallet_in_vin = wallet in vin_addrs
        wallet_in_vout = wallet in vout_addrs

        if wallet_in_vout:
            in_count += 1
            in_values.append(in_value)
            if is_coinbase:
                coinbase_in_count += 1
            for addr in vin_addrs:
                if addr and addr != wallet:
                    in_sources[addr] = in_sources.get(addr, 0) + 1

        if wallet_in_vin:
            out_count += 1
            out_values.append(out_value)
            for addr in vout_addrs:
                if addr and addr != wallet:
                    out_targets[addr] = out_targets.get(addr, 0) + 1

    tx_count = len(transactions)
    unique_in = len(in_sources)
    unique_out = len(out_targets)
    unique_counterparties = len(set(in_sources) | set(out_targets))

    in_sum = float(np.sum(in_values)) if in_values else 0.0
    out_sum = float(np.sum(out_values)) if out_values else 0.0

    in_avg = float(np.mean(in_values)) if in_values else 0.0
    out_avg = float(np.mean(out_values)) if out_values else 0.0

    in_median = float(np.median(in_values)) if in_values else 0.0
    out_median = float(np.median(out_values)) if out_values else 0.0

    in_out_count_ratio = _safe_div(in_count, out_count)
    in_out_value_ratio = _safe_div(in_sum, out_sum)

    mutual_counterparties = set(in_sources) & set(out_targets)
    reciprocity = _safe_div(len(mutual_counterparties), unique_counterparties)

    max_incoming_ratio = 0.0
    if in_count > 0 and in_sources:
        max_incoming_ratio = max(in_sources.values()) / in_count

    counterparty_entropy = _entropy(list((in_sources | out_targets).values()))

    features: dict[str, float] = {
        "tx_count": float(tx_count),
        "in_count": float(in_count),
        "out_count": float(out_count),
        "unique_in": float(unique_in),
        "unique_out": float(unique_out),
        "unique_counterparties": float(unique_counterparties),
        "in_sum": in_sum,
        "out_sum": out_sum,
        "in_avg": in_avg,
        "out_avg": out_avg,
        "in_median": in_median,
        "out_median": out_median,
        "in_out_count_ratio": in_out_count_ratio,
        "in_out_value_ratio": in_out_value_ratio,
        "reciprocity": reciprocity,
        "max_incoming_ratio": max_incoming_ratio,
        "counterparty_entropy": counterparty_entropy,
        "unique_ratio": _safe_div(unique_counterparties, tx_count),
        "coinbase_in_ratio": _safe_div(coinbase_in_count, tx_count),
    }

    return features


def extract_btc_counterparty_counts(
    transactions: list[dict[str, Any]],
    wallet_address: str,
) -> dict[str, int]:
    wallet = wallet_address.strip().lower()
    counts: dict[str, int] = {}

    for tx in transactions:
        vin_items = tx.get("vin") if isinstance(tx.get("vin"), list) else []
        vout_items = tx.get("vout") if isinstance(tx.get("vout"), list) else []

        vin_addrs: list[str] = []
        vout_addrs: list[str] = []

        for vin in vin_items:
            prevout = vin.get("prevout") if isinstance(vin, dict) else None
            if isinstance(prevout, dict):
                addr = _normalize_address(prevout.get("scriptpubkey_address"))
                if addr:
                    vin_addrs.append(addr)

        for vout in vout_items:
            if not isinstance(vout, dict):
                continue
            addr = _normalize_address(vout.get("scriptpubkey_address"))
            if addr:
                vout_addrs.append(addr)

        wallet_in_vin = wallet in vin_addrs
        wallet_in_vout = wallet in vout_addrs

        if wallet_in_vin:
            for addr in vout_addrs:
                if addr and addr != wallet:
                    counts[addr] = counts.get(addr, 0) + 1
        if wallet_in_vout:
            for addr in vin_addrs:
                if addr and addr != wallet:
                    counts[addr] = counts.get(addr, 0) + 1

    return counts


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def _normalize_address(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
