from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
for candidate in (CURRENT_DIR, CURRENT_DIR.parent):
    if (candidate / "app").exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))
        break

SCRIPTS_DIR = CURRENT_DIR / "scripts"
if not SCRIPTS_DIR.exists():
    SCRIPTS_DIR = CURRENT_DIR.parent / "scripts"
if SCRIPTS_DIR.exists() and str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from app.config.settings import load_settings
from app.core.exceptions import ExternalApiError, ValidationError
from app.data.api.explorer_client import ExplorerApiClient
from app.data.repositories.transaction_repository import TransactionRepository
from app.domain.models.network import Network
from app.services.wallet_analysis_service import WalletAnalysisService
from app.services.wallet_data_service import WalletDataService

try:
    from btc_wallet_features import (
        extract_btc_counterparty_counts,
        extract_btc_features,
        fetch_mempool_transactions,
    )
except ImportError:
    extract_btc_counterparty_counts = None
    extract_btc_features = None
    fetch_mempool_transactions = None

LABEL_ORDER = ["exchange", "individual", "miner", "mixer"]
BTC_MODEL_PATH = Path("models/babd13_btc_wallet_model.joblib")
RUS_TO_KIND = {
    "Биржа": "exchange",
    "Индивидуальный": "individual",
    "Майнер": "miner",
    "Миксер": "mixer",
}

BTC_ADDRESS_RE = re.compile(r"^(bc1[ac-hj-np-z0-9]{25,60}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})$")


def load_dataset(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    features: list[dict[str, float]] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = row.get("label")
            if label is None:
                continue
            labels.append(label)
            feat: dict[str, float] = {}
            for key, value in row.items():
                if key == "label":
                    continue
                feat[key] = safe_float(value)
            features.append(feat)

    return features, labels


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


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


def draw_graph(kind: str) -> None:
    graph = build_schematic_graph(kind)
    fig = plt.figure(figsize=(4.6, 4.0))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="#4C78A8")
    nx.draw_networkx_edges(graph, pos, width=1.2, edge_color="#777777")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white")
    plt.title(f"{kind} (schematic)")
    plt.axis("off")
    st.pyplot(fig)


def is_btc_address(address: str) -> bool:
    return bool(BTC_ADDRESS_RE.fullmatch(address.strip().lower()))


def build_btc_counterparty_graph(
    transactions: list[dict],
    wallet_address: str,
    max_nodes: int = 15,
) -> nx.Graph:
    counts = {}
    if extract_btc_counterparty_counts is not None:
        counts = extract_btc_counterparty_counts(transactions, wallet_address)

    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:max_nodes]
    graph = nx.Graph()
    graph.add_node("WALLET")
    for address, weight in top:
        label = address
        if len(address) > 12:
            label = f"{address[:6]}...{address[-4:]}"
        graph.add_edge("WALLET", label, weight=weight)
    return graph


def build_counterparty_graph(
    transactions: list[dict],
    wallet_address: str,
    max_nodes: int = 15,
) -> nx.Graph:
    wallet_lower = wallet_address.lower()
    counts: dict[str, int] = {}
    for tx in transactions:
        from_addr = str(tx.get("from", "")).lower()
        to_addr = str(tx.get("to", "")).lower()
        if not from_addr and not to_addr:
            in_msg = tx.get("in_msg") if isinstance(tx.get("in_msg"), dict) else {}
            from_addr = str(in_msg.get("source", "")).lower()
            to_addr = str(in_msg.get("destination", "")).lower()

        if not from_addr and not to_addr:
            continue

        other = to_addr if from_addr == wallet_lower else from_addr
        if not other or other == wallet_lower:
            continue
        counts[other] = counts.get(other, 0) + 1

    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:max_nodes]
    graph = nx.Graph()
    graph.add_node("WALLET")
    for address, weight in top:
        label = address
        if len(address) > 12:
            label = f"{address[:6]}...{address[-4:]}"
        graph.add_edge("WALLET", label, weight=weight)
    return graph


def draw_counterparty_graph(graph: nx.Graph, title: str) -> None:
    if graph.number_of_nodes() <= 1:
        st.info("Недостаточно транзакций для построения графа контрагентов.")
        return

    fig = plt.figure(figsize=(6.2, 4.8))
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ["#F58518" if n == "WALLET" else "#4C78A8" for n in graph.nodes()]
    node_sizes = [900 if n == "WALLET" else 500 for n in graph.nodes()]
    edge_widths = [1 + min(4, graph[u][v].get("weight", 1) / 3) for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#777777")
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white")
    plt.title(title)
    plt.axis("off")
    st.pyplot(fig)


def load_model(path: Path):
    payload = joblib.load(path)
    pipeline = payload.get("pipeline")
    if pipeline is None:
        raise ValueError("Model pipeline not found")
    return pipeline


@st.cache_resource
def init_wallet_services() -> tuple[object, WalletDataService]:
    settings = load_settings()
    api_client = ExplorerApiClient(settings)
    repository = TransactionRepository(settings.output_dir)
    analysis_service = WalletAnalysisService()
    data_service = WalletDataService(api_client, repository, analysis_service)
    return settings, data_service


def main() -> None:
    st.set_page_config(page_title="Wallet Type Demo", layout="wide")
    st.title("Wallet Type Classification Demo")

    dataset_path = Path("data/babd13_wallet4.csv")
    model_path = Path("models/babd13_wallet_type_model.joblib")

    with st.sidebar:
        st.header("Data & Model")
        if not dataset_path.exists():
            st.error("Dataset not found: data/babd13_wallet4.csv")
        if not model_path.exists():
            st.error("Model not found: models/babd13_wallet_type_model.joblib")
        if not BTC_MODEL_PATH.exists():
            st.info("BTC model not found: models/babd13_btc_wallet_model.joblib")
        st.markdown("Use the default dataset/model from this project.")

    features, labels = load_dataset(dataset_path)
    pipeline = load_model(model_path)

    tab_overview, tab_metrics, tab_predict, tab_graphs, tab_wallet = st.tabs(
        ["Overview", "Metrics", "Prediction", "Graphs", "Wallet Lookup"]
    )

    with tab_overview:
        st.write(
            "This demo shows a Random Forest classifier trained on BABD-13 features to classify wallet types (exchange, miner, mixer, individual)."
        )
        st.write("Use the Metrics tab for quality report and the Graphs tab for schematic patterns.")

    with tab_metrics:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
        st.subheader("Confusion Matrix")
        st.write(cm)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, digits=3, output_dict=True)
        st.dataframe(report)

    with tab_predict:
        st.subheader("Random sample prediction")
        idx = st.slider("Sample index", 0, len(features) - 1, 0)
        sample = features[idx]
        probs = pipeline.predict_proba([sample])[0]
        pred = pipeline.classes_[int(np.argmax(probs))]
        st.write(f"Predicted label: {pred}")
        st.write("Probabilities:")
        st.json({label: float(prob) for label, prob in zip(pipeline.classes_, probs)})

    with tab_graphs:
        st.subheader("Schematic transaction graphs")
        choice = st.selectbox("Select class", LABEL_ORDER, index=0)
        draw_graph(choice)

    with tab_wallet:
        settings, data_service = init_wallet_services()
        st.subheader("Wallet lookup")
        st.caption("API ключи берутся из .env. Вводить их в интерфейс не нужно.")

        address = st.text_input("Адрес кошелька (BTC / EVM / TON)")
        is_btc = bool(address) and is_btc_address(address)

        auto_detect = None
        selected_network = None
        if not is_btc:
            auto_detect = st.checkbox(
                "Авто-определение сети",
                value=bool(getattr(settings, "auto_detect_network", True)),
            )
            if not auto_detect:
                selected_network = st.selectbox(
                    "Сеть",
                    list(Network),
                    format_func=lambda item: item.ui_label,
                )

        if st.button("Определить тип"):
            if not address:
                st.warning("Введите адрес кошелька.")
            else:
                try:
                    if is_btc:
                        if fetch_mempool_transactions is None or extract_btc_features is None:
                            st.error("BTC helpers not available. Ensure scripts/btc_wallet_features.py exists.")
                            return
                        if not BTC_MODEL_PATH.exists():
                            st.error(
                                "BTC model not found. Run scripts/prepare_babd13_btc_dataset.py and train the model."
                            )
                            return

                        with st.spinner("Загрузка BTC транзакций..."):
                            transactions = fetch_mempool_transactions(
                                address,
                                max_tx=settings.max_transactions,
                                request_timeout=settings.request_timeout_seconds,
                            )

                        if not transactions:
                            st.warning("Нет транзакций по адресу.")
                            return

                        features = extract_btc_features(transactions, address)
                        btc_pipeline = load_model(BTC_MODEL_PATH)
                        probs = btc_pipeline.predict_proba([features])[0]
                        pred = btc_pipeline.classes_[int(np.argmax(probs))]

                        st.success("BTC анализ завершен.")
                        st.write(f"Тип кошелька: {pred}")
                        st.write(f"Транзакций: {len(transactions)}")
                        st.write("Вероятности:")
                        st.json({label: float(prob) for label, prob in zip(btc_pipeline.classes_, probs)})

                        st.subheader("Граф контрагентов")
                        graph = build_btc_counterparty_graph(transactions, address)
                        draw_counterparty_graph(graph, "Контрагенты (top 15)")

                        if pred in LABEL_ORDER:
                            st.subheader("Схематический граф класса")
                            draw_graph(pred)
                    else:
                        with st.spinner("Загрузка транзакций..."):
                            network = (
                                data_service.detect_network(address)
                                if auto_detect
                                else selected_network
                            )
                            transactions = data_service.fetch_transactions(address, network)
                            saved_paths = data_service.save_raw_transactions(
                                address, network, transactions
                            )
                            result = data_service.build_result(
                                address, network, transactions, saved_paths
                            )

                        st.success("Анализ завершен.")
                        st.write(f"Сеть: {result.network.ui_label}")
                        st.write(f"Тип кошелька: {result.wallet_type}")
                        st.write(f"Транзакций: {result.transaction_count}")
                        st.write(f"Риск: {result.risk_level} (score {result.risk_score})")

                        if result.category_stats:
                            st.subheader("Статистика категорий")
                            st.json(result.category_stats)

                        if result.portrait:
                            st.subheader("Краткий портрет")
                            st.write(result.portrait.summary)

                        st.subheader("Граф контрагентов")
                        graph = build_counterparty_graph(transactions, address)
                        draw_counterparty_graph(graph, "Контрагенты (top 15)")

                        kind = RUS_TO_KIND.get(result.wallet_type)
                        if kind:
                            st.subheader("Схематический граф класса")
                            draw_graph(kind)
                except (ExternalApiError, ValidationError) as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
