from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


LABEL_ORDER = ["exchange", "individual", "miner", "mixer"]


def load_dataset(path: Path, needed_cols: list[str]) -> tuple[list[dict[str, float]], list[str], dict[str, list[float]]]:
    features: list[dict[str, float]] = []
    labels: list[str] = []
    collected: dict[str, list[float]] = {name: [] for name in needed_cols}

    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Dataset CSV has no header")

        for row in reader:
            label = row.get("label")
            if label is None:
                continue
            labels.append(label)

            feature_row: dict[str, float] = {}
            for key, value in row.items():
                if key == "label":
                    continue
                feature_row[key] = safe_float(value)
                if key in collected:
                    collected[key].append(feature_row[key])
            features.append(feature_row)

    return features, labels, collected


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def choose_column(fieldnames: list[str], preferred: str, prefix: str) -> str:
    if preferred in fieldnames:
        return preferred
    for name in fieldnames:
        if name.startswith(prefix):
            return name
    for name in fieldnames:
        if name != "label":
            return name
    return preferred


def plot_box_by_label(values: list[float], labels: list[str], title: str, out_path: Path) -> None:
    data = []
    for label in LABEL_ORDER:
        data.append([v for v, y in zip(values, labels) if y == label])

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, tick_labels=LABEL_ORDER, showfliers=False)
    plt.title(title)
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=30, ha="right")
    plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(model_path: Path, out_path: Path, top_n: int = 15) -> None:
    payload = joblib.load(model_path)
    pipeline = payload.get("pipeline")
    if pipeline is None:
        raise ValueError("Model pipeline not found")

    vectorizer = pipeline.named_steps["vectorizer"]
    model = pipeline.named_steps["model"]

    names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    selected_names = [names[i] for i in order][::-1]
    selected_scores = [importances[i] for i in order][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(selected_names, selected_scores)
    plt.xlabel("importance")
    plt.title("Random Forest feature importance (top 15)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_schematic_graph(kind: str) -> nx.Graph:
    graph = nx.Graph()
    if kind == "exchange":
        center = "EX"
        graph.add_node(center)
        for i in range(1, 16):
            graph.add_edge(center, f"U{i}")
        return graph

    if kind == "miner":
        pool = "POOL"
        graph.add_node(pool)
        for i in range(1, 10):
            graph.add_edge(pool, f"M{i}")
        return graph

    if kind == "mixer":
        mixer = "MIX"
        graph.add_node(mixer)
        for i in range(1, 8):
            graph.add_edge(f"IN{i}", mixer)
        for i in range(1, 10):
            graph.add_edge(mixer, f"OUT{i}")
        return graph

    if kind == "individual":
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("C", "D")
        return graph

    return graph


def plot_schematic_graph(kind: str, out_path: Path, title: str) -> None:
    graph = build_schematic_graph(kind)
    plt.figure(figsize=(4.8, 4.2))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="#4C78A8")
    nx.draw_networkx_edges(graph, pos, width=1.2, edge_color="#777777")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/babd13_wallet4.csv")
    parser.add_argument("--model", default="models/babd13_wallet_type_model.joblib")
    parser.add_argument("--outdir", default="outputs/figures")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Dataset CSV has no header")
        fieldnames = reader.fieldnames

    degree_col = choose_column(fieldnames, preferred="PDIa1-1", prefix="PDI")
    diversity_col = choose_column(fieldnames, preferred="CI3a22-5", prefix="CI")

    features, labels, collected = load_dataset(dataset_path, [degree_col, diversity_col])

    plot_box_by_label(
        collected[degree_col],
        labels,
        title=f"Degree proxy by class ({degree_col})",
        out_path=outdir / "fig_2_1_degree_distribution.png",
    )

    plot_box_by_label(
        collected[diversity_col],
        labels,
        title=f"Counterparty diversity proxy ({diversity_col})",
        out_path=outdir / "fig_2_2_diversity_distribution.png",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    payload = joblib.load(model_path)
    pipeline = payload.get("pipeline")
    if pipeline is None:
        raise ValueError("Model pipeline not found")

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    plot_confusion(cm, outdir / "fig_2_3_confusion_matrix.png")

    plot_feature_importance(model_path, outdir / "fig_3_1_feature_importance.png")

    plot_schematic_graph(
        "exchange",
        outdir / "fig_3_2_exchange_graph.png",
        "Exchange address (schematic)",
    )
    plot_schematic_graph(
        "miner",
        outdir / "fig_3_3_miner_graph.png",
        "Miner address (schematic)",
    )
    plot_schematic_graph(
        "mixer",
        outdir / "fig_3_4_mixer_graph.png",
        "Mixer address (schematic)",
    )
    plot_schematic_graph(
        "individual",
        outdir / "fig_3_5_individual_graph.png",
        "Individual address (schematic)",
    )

    print("Saved figures to:", outdir)


if __name__ == "__main__":
    main()
