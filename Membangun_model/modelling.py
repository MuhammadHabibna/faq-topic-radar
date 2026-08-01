"""
modelling.py
Kriteria 2 - Basic: fit BERTopic SEKALI menggunakan BEST_CONFIG hasil riset
Colab (lihat Eksperimen_Topic_Modeling_FAQ.ipynb, Section 6), TANPA
hyperparameter tuning ulang.

Jalankan dari folder Membangun_model/:
    python modelling.py
"""

import os

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
MODEL_OUTPUT_DIR = "bertopic_model_final"
RANDOM_SEED = 42

# Hasil riset Colab (180 kombinasi, 3 model embedding x 60 kombinasi UMAP/HDBSCAN)
# composite_score tertinggi: 0.6896
BEST_CONFIG = {
    "embedding_model_name": "BAAI/bge-small-en-v1.5",
    "n_neighbors": 20,
    "n_components": 10,
    "min_dist": 0.10,
    "min_cluster_size": 75,
    "min_samples": None,
    "cluster_selection_method": "eom",
}


def purity_score(y_true, y_pred) -> float:
    contingency = pd.crosstab(y_pred, y_true)
    return contingency.max(axis=1).sum() / contingency.values.sum()


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment("topic_modeling_basic")


def fit_and_evaluate(df: pd.DataFrame, config: dict):
    embedding_model = SentenceTransformer(config["embedding_model_name"])
    embeddings = embedding_model.encode(df["utterance"].tolist(), show_progress_bar=True)

    umap_model = umap.UMAP(
        n_neighbors=config["n_neighbors"],
        n_components=config["n_components"],
        min_dist=config["min_dist"],
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=config["min_cluster_size"],
        min_samples=config["min_samples"],
        cluster_selection_method=config["cluster_selection_method"],
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(df["utterance"].tolist(), embeddings=embeddings)
    topics = np.array(topics)

    metrics = {
        "nmi": normalized_mutual_info_score(df["category"], topics),
        "ari": adjusted_rand_score(df["category"], topics),
        "purity": purity_score(df["category"], topics),
        "outlier_ratio": (topics == -1).sum() / len(topics),
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
    }
    return topic_model, metrics


def main():
    setup_mlflow()

    df = pd.read_csv(DATA_PATH)
    print(f"Data dimuat: {len(df)} baris")

    with mlflow.start_run(run_name="best_config_single_fit"):
        mlflow.log_params(BEST_CONFIG)

        topic_model, metrics = fit_and_evaluate(df, BEST_CONFIG)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        topic_model.save(
            MODEL_OUTPUT_DIR,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=True,
        )
        mlflow.log_artifacts(MODEL_OUTPUT_DIR, artifact_path=MODEL_OUTPUT_DIR)

        print("\nHasil fit BEST_CONFIG:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
        print(f"\nModel tersimpan & ter-log ke MLflow (experiment: topic_modeling_basic)")


if __name__ == "__main__":
    main()
