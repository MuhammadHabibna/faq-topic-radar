"""
modelling_tuning.py
Kriteria 2 - Skilled/Advance: hyperparameter tuning ekstensif tapi dipangkas
jadi 10 kombinasi representatif (BUKAN full 180 kombinasi dari Colab, yang
makan 1 jam+). 10 kombinasi ini diambil LANGSUNG dari leaderboard_df.csv
hasil riset Colab, sengaja dicampur:
  - 1 pemenang (composite_score tertinggi)
  - 4 saingan terdekat
  - 2 medioker
  - 3 sengaja jelek (semua min_cluster_size=15) - buat bukti visual di MLflow
    kenapa min_cluster_size kekecilan bikin data kebelah jadi ratusan topik
    noise (ARI ambruk).

Semua kombinasi pakai embedding bge_small (BAAI/bge-small-en-v1.5) — model
embedding sendiri TIDAK ikut di-tuning ulang di sini, karena perbandingan
3 model embedding sudah kelar di riset Colab (hasilnya nyaris identik,
lihat comparison_df di ringkasan_eksperimen.txt).

Tiap kombinasi di-log sebagai 1 run TERPISAH ke MLflow (manual logging,
bukan autolog) — biar histori percobaannya keliatan permanen & bisa
dibandingin di DagsHub.

Jalankan dari folder Membangun_model/:
    python modelling_tuning.py
"""

import os
import time

import numpy as np
import pandas as pd
import mlflow
import umap
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"

DATA_PATH = "faq_dataset_preprocessing/cleaned_with_category.csv"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RANDOM_SEED = 42

# 10 kombinasi representatif, diambil dari leaderboard_df.csv hasil riset Colab
# (baris 166, 147, 136, 145, 161, 137, 168, 178, 152, 179)
HYPERPARAMETER_COMBINATIONS = [
    {"label": "pemenang",       "n_neighbors": 20, "n_components": 10, "min_dist": 0.10, "min_cluster_size": 75,  "min_samples": None, "cluster_selection_method": "eom"},
    {"label": "saingan_dekat_1","n_neighbors": 15, "n_components": 5,  "min_dist": 0.00, "min_cluster_size": 100, "min_samples": None, "cluster_selection_method": "eom"},
    {"label": "saingan_dekat_2","n_neighbors": 20, "n_components": 5,  "min_dist": 0.10, "min_cluster_size": 100, "min_samples": None, "cluster_selection_method": "leaf"},
    {"label": "saingan_dekat_3","n_neighbors": 50, "n_components": 15, "min_dist": 0.00, "min_cluster_size": 100, "min_samples": 10,   "cluster_selection_method": "eom"},
    {"label": "saingan_dekat_4","n_neighbors": 20, "n_components": 15, "min_dist": 0.05, "min_cluster_size": 100, "min_samples": None, "cluster_selection_method": "leaf"},
    {"label": "medioker_1",     "n_neighbors": 10, "n_components": 5,  "min_dist": 0.10, "min_cluster_size": 50,  "min_samples": 10,   "cluster_selection_method": "eom"},
    {"label": "medioker_2",     "n_neighbors": 10, "n_components": 15, "min_dist": 0.00, "min_cluster_size": 75,  "min_samples": 10,   "cluster_selection_method": "leaf"},
    {"label": "sengaja_jelek_1","n_neighbors": 10, "n_components": 15, "min_dist": 0.05, "min_cluster_size": 15,  "min_samples": 10,   "cluster_selection_method": "eom"},
    {"label": "sengaja_jelek_2","n_neighbors": 20, "n_components": 5,  "min_dist": 0.00, "min_cluster_size": 15,  "min_samples": None, "cluster_selection_method": "leaf"},
    {"label": "sengaja_jelek_3","n_neighbors": 30, "n_components": 15, "min_dist": 0.00, "min_cluster_size": 15,  "min_samples": 5,    "cluster_selection_method": "leaf"},
]


def purity_score(y_true, y_pred) -> float:
    contingency = pd.crosstab(y_pred, y_true)
    return contingency.max(axis=1).sum() / contingency.values.sum()


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment("topic_modeling_tuning")


def compute_composite_score(nmi: float, ari: float, outlier_ratio: float, n_topics: int) -> float:
    if n_topics < 2:
        return 0.0
    return 0.4 * nmi + 0.3 * ari + 0.3 * (1 - outlier_ratio)


def evaluate_combo(embeddings: np.ndarray, categories, combo: dict):
    umap_model = umap.UMAP(
        n_neighbors=combo["n_neighbors"],
        n_components=combo["n_components"],
        min_dist=combo["min_dist"],
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=combo["min_cluster_size"],
        min_samples=combo["min_samples"],
        cluster_selection_method=combo["cluster_selection_method"],
        metric="euclidean",
        prediction_data=True,
    )

    reduced = umap_model.fit_transform(embeddings)
    labels = hdbscan_model.fit_predict(reduced)

    outlier_ratio = (labels == -1).sum() / len(labels)
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    nmi = normalized_mutual_info_score(categories, labels)
    ari = adjusted_rand_score(categories, labels)
    purity = purity_score(categories, labels)
    composite = compute_composite_score(nmi, ari, outlier_ratio, n_topics)

    return {
        "nmi": nmi,
        "ari": ari,
        "purity": purity,
        "outlier_ratio": outlier_ratio,
        "n_topics": n_topics,
        "composite_score": composite,
    }


def main():
    setup_mlflow()

    df = pd.read_csv(DATA_PATH)
    texts = df["utterance"].tolist()
    categories = df["category"].values
    print(f"Data dimuat: {len(df)} baris")

    print(f"Generate embedding sekali pakai {EMBEDDING_MODEL_NAME} (dipakai ulang di semua kombinasi)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    results = []
    start_time = time.time()

    for i, combo in enumerate(HYPERPARAMETER_COMBINATIONS, start=1):
        label = combo["label"]
        hyperparams = {k: v for k, v in combo.items() if k != "label"}

        print(f"\n[{i}/{len(HYPERPARAMETER_COMBINATIONS)}] Kombinasi '{label}': {hyperparams}")

        with mlflow.start_run(run_name=f"combo_{i:02d}_{label}"):
            mlflow.log_param("embedding_model_name", EMBEDDING_MODEL_NAME)
            mlflow.log_params(hyperparams)
            mlflow.set_tag("kategori_kombinasi", label)

            metrics = evaluate_combo(embeddings, categories, hyperparams)

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            results.append({"label": label, **hyperparams, **metrics})
            print(f"  -> NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, "
                  f"outlier={metrics['outlier_ratio']:.2%}, n_topics={metrics['n_topics']}, "
                  f"composite={metrics['composite_score']:.4f}")

    total_minutes = (time.time() - start_time) / 60
    print(f"\nSelesai {len(HYPERPARAMETER_COMBINATIONS)} kombinasi dalam {total_minutes:.1f} menit")

    results_df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    print("\nRingkasan seluruh kombinasi (diurutkan composite_score):")
    print(results_df[["label", "nmi", "ari", "outlier_ratio", "n_topics", "composite_score"]].to_string(index=False))

    best_label = results_df.iloc[0]["label"]
    print(f"\nKombinasi terbaik dari tuning ini: '{best_label}' "
          f"(composite_score={results_df.iloc[0]['composite_score']:.4f})")
    print("Kalau hasilnya konsisten sama riset Colab, ini harusnya 'pemenang' - "
          "mengonfirmasi BEST_CONFIG di modelling.py masih valid.")


if __name__ == "__main__":
    main()
