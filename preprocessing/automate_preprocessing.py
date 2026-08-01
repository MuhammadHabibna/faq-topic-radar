"""
automate_preprocessing.py
Preprocessing otomatis: baca seluruh CSV di faq_dataset_raw/ (akumulatif),
bersihkan, generate embedding. Jalankan dari folder preprocessing/:
    python automate_preprocessing.py
"""

import os
import re
import glob
import unicodedata

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import dagshub
import mlflow

RAW_DIR = "../faq_dataset_raw"
OUTPUT_DIR = "faq_dataset_preprocessing"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_WORDS = 3

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"


def log_to_mlflow(initial_rows: int, df_clean: pd.DataFrame) -> None:
    """Log parameter & metric preprocessing ke MLflow (tracking server DagsHub)."""
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("data_preparation")

    with mlflow.start_run():
        mlflow.log_param("initial_rows", initial_rows)
        mlflow.log_param("min_word_threshold", MIN_WORDS)
        mlflow.log_metric("final_row_count", len(df_clean))
        mlflow.log_metric("rows_dropped", initial_rows - len(df_clean))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "cleaned_text.csv"))
    print("Logged ke MLflow (DagsHub)")


def load_data(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Baca SEMUA file CSV di raw_dir, gabung jadi 1 dataframe, ambil
    kolom 'utterance' aja (konteks real: cuma teks pertanyaan user)."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Gak ada file CSV ditemukan di {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df = df[["utterance"]].copy()
    print(f"[load_data] {len(csv_files)} file dibaca, total {len(df)} baris")
    return df


def _light_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(df: pd.DataFrame, min_words: int = MIN_WORDS) -> pd.DataFrame:
    """Buang teks kependekan + near-duplicate, rapikan unicode/whitespace.
    TIDAK stopword removal/stemming — sentence-transformer butuh kalimat utuh."""
    df = df.copy()

    before = len(df)
    df["n_words"] = df["utterance"].str.split().str.len()
    df = df[df["n_words"] >= min_words].drop(columns=["n_words"])
    print(f"[clean_text] Buang teks < {min_words} kata: {before} -> {len(df)}")

    before = len(df)
    key = df["utterance"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.loc[~key.duplicated(keep="first")]
    print(f"[clean_text] Buang near-duplicate: {before} -> {len(df)}")

    df["utterance"] = df["utterance"].apply(_light_clean)
    return df.reset_index(drop=True)


def generate_embeddings(df: pd.DataFrame, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Generate sentence embedding pakai model pretrained (beku, gak ditraining)."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["utterance"].tolist(), show_progress_bar=True, batch_size=64)
    print(f"[generate_embeddings] Shape: {embeddings.shape}")
    return embeddings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = load_data()
    df_clean = clean_text(df_raw)
    embeddings = generate_embeddings(df_clean)

    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_text.csv")
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    df_clean.to_csv(cleaned_path, index=False)
    np.save(embeddings_path, embeddings)

    print(f"\nDisimpan ke: {cleaned_path}")
    print(f"Disimpan ke: {embeddings_path}")

    log_to_mlflow(len(df_raw), df_clean)


if __name__ == "__main__":
    main()
