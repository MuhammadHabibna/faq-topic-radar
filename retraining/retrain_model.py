"""
retrain_model.py
Kriteria 3: Continuous retraining mingguan. Refit BERTopic dari seluruh data
terakumulasi (lama + baru), pakai config hyperparameter yang SUDAH DIPUTUSKAN
(diambil otomatis dari MLflow Model Registry -- TIDAK ada tuning ulang di
sini). Topik hasil refit di-reconcile sama topik dari model Production
minggu lalu (biar topik yang isinya sama gak dianggap baru tiap minggu),
topik yang BENAR-BENAR baru dilabel lewat Gemini API, semuanya di-log ke
MLflow (DagsHub).

Model baru di-register ke Model Registry tapi TIDAK otomatis dipromote ke
Production -- itu keputusan manual setelah review metrik.

Jalankan dari folder retraining/:
    python retrain_model.py

CATATAN WINDOWS: jalankan dengan $env:PYTHONIOENCODING="utf-8" dulu (PowerShell)
atau set PYTHONIOENCODING=utf-8 (CMD) sebelum run -- MLflow print emoji ke
stdout yang gak kebaca default codepage Windows (cp1252).
"""

import json
import os
import random
import re
import time
from datetime import date

import google.generativeai as genai
import hdbscan
import mlflow
import numpy as np
import pandas as pd
import umap
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from google.api_core.exceptions import ResourceExhausted
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"
MODEL_REGISTRY_NAME = "faq-topic-model"
MODEL_ARTIFACT_DIR = "bertopic_model_final"
TOPIC_EMBEDDINGS_FILENAME = "topic_embeddings.safetensors"
BASELINE_LABELS_FILENAME = "topic_labels_v1_baseline.json"

EMBEDDINGS_PATH = "../preprocessing/faq_dataset_preprocessing/embeddings.npy"
CLEANED_TEXT_PATH = "../preprocessing/faq_dataset_preprocessing/cleaned_text.csv"

RANDOM_SEED = 42

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_MIN_INTERVAL_SECONDS = 4.5
GEMINI_MAX_RETRIES = 5
GEMINI_CHECKPOINT_PATH = "gemini_label_checkpoint.json"

COHERENCE_SAMPLE_SIZE = 3000

# Threshold cosine similarity buat nganggep topik lama & topik baru "topik
# yang sama". Ini ASUMSI AWAL (belum divalidasi empiris lewat review manual
# beberapa minggu pertama retraining) -- BUKAN angka final mutlak, cuma
# starting point yang wajar buat embedding bge_small.
TOPIC_SIMILARITY_THRESHOLD = 0.75

# Config fallback kalau ini run PERTAMA (belum ada versi Production sama
# sekali) -- hasil riset Colab, sama seperti BEST_CONFIG di modelling.py.
BOOTSTRAP_CONFIG = {
    "embedding_model_name": "BAAI/bge-small-en-v1.5",
    "n_neighbors": 20,
    "n_components": 10,
    "min_dist": 0.10,
    "min_cluster_size": 75,
    "min_samples": None,
    "cluster_selection_method": "eom",
}

_TMP_DIR = "_retrain_tmp"
_TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


# ============================================================
# Helper kecil (bukan salah satu dari 8 tahap utama, cuma plumbing)
# ============================================================

def _connect_mlflow() -> None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")


def _load_accumulated_data() -> tuple[np.ndarray, list[str]]:
    print(f"Load {EMBEDDINGS_PATH} & {CLEANED_TEXT_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CLEANED_TEXT_PATH)
    texts = df["utterance"].tolist()
    if len(texts) != len(embeddings):
        raise ValueError(
            f"Jumlah baris cleaned_text.csv ({len(texts)}) != jumlah embeddings.npy "
            f"({len(embeddings)}) -- pastikan dua file ini hasil automate_preprocessing.py "
            "run yang SAMA."
        )
    print(f"  -> {len(texts)} dokumen, embedding dim {embeddings.shape[1]}.")
    return embeddings, texts


def _tokenize_simple(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


# ============================================================
# Fungsi 1 -- Ambil config aktif dari MLflow Model Registry
# ============================================================

def _parse_config_params(raw_params: dict) -> dict:
    """MLflow nyimpen semua param sebagai string -- parse ulang ke tipe asli."""
    raw_min_samples = raw_params.get("min_samples")
    min_samples = None if raw_min_samples in (None, "None") else int(raw_min_samples)
    return {
        "embedding_model_name": raw_params["embedding_model_name"],
        "n_neighbors": int(raw_params["n_neighbors"]),
        "n_components": int(raw_params["n_components"]),
        "min_dist": float(raw_params["min_dist"]),
        "min_cluster_size": int(raw_params["min_cluster_size"]),
        "min_samples": min_samples,
        "cluster_selection_method": raw_params["cluster_selection_method"],
    }


def get_production_config() -> tuple[dict, bool]:
    print("Connect ke MLflow tracking server (DagsHub)...")
    _connect_mlflow()
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    except MlflowException:
        versions = []

    if not versions:
        print(f"Belum ada versi '{MODEL_REGISTRY_NAME}' di stage Production -- bootstrap run pertama.")
        print("Pakai config hardcode hasil riset Colab (BOOTSTRAP_CONFIG):")
        for key, value in BOOTSTRAP_CONFIG.items():
            print(f"  {key}: {value}")
        return dict(BOOTSTRAP_CONFIG), True

    production_version = versions[0]
    run = client.get_run(production_version.run_id)
    config = _parse_config_params(run.data.params)
    print(f"Config diambil dari '{MODEL_REGISTRY_NAME}' versi {production_version.version} "
          f"(run_id={production_version.run_id}):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return config, False


# ============================================================
# Fungsi 2 -- Ambil data topik lama (BUKAN load seluruh model)
# ============================================================

def _download_latest_topic_assignments_csv(client: MlflowClient, run_id: str) -> str | None:
    artifacts = client.list_artifacts(run_id)
    candidates = sorted(
        a.path for a in artifacts if a.path.startswith("topic_assignments_") and a.path.endswith(".csv")
    )
    if not candidates:
        return None
    # nama file ada tanggal ISO-nya (topic_assignments_YYYY-MM-DD.csv), jadi
    # sort string = sort waktu -> ambil yang paling baru
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=candidates[-1])


def load_previous_model(is_bootstrap: bool) -> dict | None:
    """Ambil 2 hal ringan dari run Production minggu lalu (BUKAN load
    BERTopic penuh -- gak butuh UMAP/HDBSCAN/embedding model-nya lagi,
    karena model baru fit componentnya sendiri dari nol):
      1. topic_embeddings.safetensors -> centroid tiap topik lama
      2. CSV topic_assignments lama, dedupe -> {topic_id: gemini_label}
    """
    print("Ambil data topik dari run Production minggu lalu...")
    if is_bootstrap:
        print("  -> Bootstrap run, belum ada model sebelumnya. Skip.")
        return None

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    run_id = versions[0].run_id

    print(f"  -> Download {TOPIC_EMBEDDINGS_FILENAME} dari run {run_id}...")
    safetensors_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"{MODEL_ARTIFACT_DIR}/{TOPIC_EMBEDDINGS_FILENAME}"
    )
    old_embeddings = load_file(safetensors_path)["topic_embeddings"]

    csv_path = _download_latest_topic_assignments_csv(client, run_id)
    if csv_path is None:
        # Run Production ini belum pernah lewat retrain_model.py (kemungkinan
        # masih model baseline Kriteria 2) -- gak ada CSV topic_assignments,
        # jadi gak ada label lama. Baris safetensors-nya urut by raw topic id
        # BERTopic (0..N-1, besar->kecil by ukuran), treat langsung sebagai
        # final id "generasi pertama". Perlu topics.json (file kecil, bukan
        # model penuh) buat tau berapa baris outlier di depan (_outliers).
        print(f"  -> Gak ketemu topic_assignments_*.csv di run {run_id} (run baseline, belum "
              "pernah lewat reconciliation).")
        topics_json_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{MODEL_ARTIFACT_DIR}/topics.json"
        )
        with open(topics_json_path, "r", encoding="utf-8") as f:
            offset = int(json.load(f)["_outliers"])
        n_real = max(len(old_embeddings) - offset, 0)
        real_rows = old_embeddings[offset:]
        centroids_by_final_id = {i: real_rows[i] for i in range(n_real)}

        # Cek ada gak backfill label baseline (hasil backfill_v1_labels.py,
        # one-off) sebelum nyerah total ke labels kosong.
        try:
            baseline_labels_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=BASELINE_LABELS_FILENAME
            )
            with open(baseline_labels_path, "r", encoding="utf-8") as f:
                labels = {int(k): v for k, v in json.load(f).items()}
            print(f"  -> Ketemu label baseline v1 ({BASELINE_LABELS_FILENAME}), "
                  f"{len(labels)} topik punya label.")
        except Exception:
            labels = {}
            print("  -> Tidak ada label baseline sama sekali, lanjut tanpa label lama.")
    else:
        old_csv = pd.read_csv(csv_path)
        old_real = old_csv[old_csv["topic_id"] != -1]
        labels = old_real.drop_duplicates("topic_id").set_index("topic_id")["topic_label"].to_dict()

        # Invariant BERTopic: topic_embeddings_ SELALU terurut besar->kecil by
        # jumlah dokumen (raw id 0 = topik terbesar). Jadi rank ukuran topik
        # di CSV minggu lalu = urutan baris di safetensors minggu lalu --
        # inilah cara kita "menerjemahkan" baris safetensors (anonim) jadi
        # final_topic_id tanpa perlu artifact ke-3.
        size_rank = old_real.groupby("topic_id").size().sort_values(ascending=False)
        n_real = len(size_rank)
        real_rows = old_embeddings[-n_real:] if n_real else old_embeddings[:0]
        centroids_by_final_id = {
            final_id: real_rows[rank] for rank, final_id in enumerate(size_rank.index)
        }

    print(f"  -> {len(centroids_by_final_id)} topik lama ter-load ({len(labels)} punya label).")
    return {"centroids": centroids_by_final_id, "labels": labels}


# ============================================================
# Fungsi 3 -- Fit model baru (pola identik modelling.py)
# ============================================================

def fit_new_model(embeddings: np.ndarray, texts: list[str], config: dict) -> tuple[BERTopic, np.ndarray]:
    print(f"Fit model baru pakai embedding model '{config['embedding_model_name']}'...")
    embedding_model = SentenceTransformer(config["embedding_model_name"])

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
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    topics = np.array(topics)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  -> Selesai. {n_topics} topik, outlier {int((topics == -1).sum())}/{len(topics)}.")
    return topic_model, topics


# ============================================================
# Fungsi 4 -- Reconcile topik baru vs topik lama
# ============================================================

def _get_new_topic_centroids(topic_model: BERTopic) -> dict[int, np.ndarray]:
    offset = topic_model._outliers
    topic_ids = [t for t in topic_model.get_topic_info()["Topic"].tolist() if t != -1]
    return {tid: topic_model.topic_embeddings_[tid + offset] for tid in topic_ids}


def _print_labeling_estimate(n_new_topics: int) -> None:
    if n_new_topics == 0:
        return
    estimasi_menit = n_new_topics * GEMINI_MIN_INTERVAL_SECONDS / 60
    print(f"  -> Estimasi waktu labeling Gemini: {n_new_topics} topik x ~{GEMINI_MIN_INTERVAL_SECONDS}s "
          f"= ~{estimasi_menit:.1f} menit.")


def reconcile_topics(
    old_topics: dict | None,
    new_model: BERTopic,
    similarity_threshold: float = TOPIC_SIMILARITY_THRESHOLD,
) -> dict:
    print("Reconcile topik baru vs topik lama...")
    new_centroids = _get_new_topic_centroids(new_model)
    new_topic_ids = sorted(new_centroids.keys())

    if old_topics is None:
        print(f"  -> Gak ada topik lama (bootstrap). Semua {len(new_topic_ids)} topik ditandai BARU.")
        _print_labeling_estimate(len(new_topic_ids))
        return {
            new_id: {"final_topic_id": new_id, "topic_label": None, "is_new_topic": True, "similarity_score": None}
            for new_id in new_topic_ids
        }

    old_centroids = old_topics["centroids"]
    old_labels = old_topics["labels"]
    old_final_ids = sorted(old_centroids.keys())

    old_matrix = np.array([old_centroids[t] for t in old_final_ids])
    new_matrix = np.array([new_centroids[t] for t in new_topic_ids])
    sim_matrix = cosine_similarity(old_matrix, new_matrix)

    # Greedy one-to-one matching: pasangan similarity tertinggi menang duluan.
    pairs = sorted(
        (
            (sim_matrix[i, j], old_id, new_id)
            for i, old_id in enumerate(old_final_ids)
            for j, new_id in enumerate(new_topic_ids)
        ),
        key=lambda p: p[0],
        reverse=True,
    )

    matched_old: set[int] = set()
    matched_new: set[int] = set()
    new_to_old: dict[int, tuple[int, float]] = {}
    for sim, old_id, new_id in pairs:
        if sim < similarity_threshold:
            break  # udah terurut desc, sisanya pasti di bawah threshold juga
        if old_id in matched_old or new_id in matched_new:
            continue
        matched_old.add(old_id)
        matched_new.add(new_id)
        new_to_old[new_id] = (old_id, float(sim))

    # Topik genuinely-baru butuh id GLOBAL yang belum pernah dipakai generasi
    # manapun sebelumnya -- kalau reuse raw id new_model (yang tiap minggu
    # restart dari 0), bisa tabrakan sama final_topic_id topik lain dari
    # minggu-minggu sebelumnya.
    next_new_id = max(old_final_ids) + 1

    reconciliation_map = {}
    for new_id in new_topic_ids:
        if new_id in new_to_old:
            old_id, sim = new_to_old[new_id]
            reconciliation_map[new_id] = {
                "final_topic_id": old_id,
                "topic_label": old_labels.get(old_id),
                "is_new_topic": False,
                "similarity_score": sim,
            }
        else:
            reconciliation_map[new_id] = {
                "final_topic_id": next_new_id,
                "topic_label": None,
                "is_new_topic": True,
                "similarity_score": None,
            }
            next_new_id += 1

    n_matched = len(matched_new)
    n_new = len(new_topic_ids) - n_matched
    print(f"  -> {n_matched} topik matched (reuse label lama), {n_new} topik baru (perlu label Gemini).")
    _print_labeling_estimate(n_new)
    return reconciliation_map


# ============================================================
# Fungsi 5 -- Label topik baru pakai Gemini (rate-limit safe)
# ============================================================

def _load_label_checkpoint() -> dict[int, str]:
    if os.path.exists(GEMINI_CHECKPOINT_PATH):
        with open(GEMINI_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  -> Checkpoint lama ketemu: {len(data)} topik udah pernah kelabel, di-skip.")
        return {int(k): v for k, v in data.items()}
    return {}


def _save_label_checkpoint(checkpoint: dict[int, str]) -> None:
    with open(GEMINI_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


def _build_labeling_prompt(top_words: list[str], sample_docs: list[str]) -> str:
    words_str = ", ".join(top_words)
    docs_str = "\n".join(f"- {doc}" for doc in sample_docs)
    return f"""You are a topic modeling expert. Based on the topic info below, give ONE
short, human-readable label for this topic (example: "Payment & Billing Issues").

Top keywords (c-TF-IDF): {words_str}

Most representative example documents:
{docs_str}

Answer with ONLY the short label (max 5 words). The label MUST be in English,
regardless of the language of the keywords/documents above. No extra explanation."""


def _is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, ResourceExhausted):
        return True
    message = str(e).lower()
    return "429" in message or "quota" in message or "rate limit" in message


def _call_gemini_with_retry(gemini_model, prompt: str, final_id: int) -> str:
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if not _is_rate_limit_error(e):
                print(f"    GAGAL labeling topik {final_id} ({type(e).__name__}: {e})")
                raise
            wait = GEMINI_MIN_INTERVAL_SECONDS * (2 ** (attempt - 1))
            print(f"    Rate limit (429) buat topik {final_id}, percobaan {attempt}/{GEMINI_MAX_RETRIES}. "
                  f"Nunggu {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Gagal labeling topik {final_id} setelah {GEMINI_MAX_RETRIES}x percobaan (rate limit terus).")


def label_new_topics_with_gemini(new_model: BERTopic, reconciliation_map: dict) -> dict:
    new_entries = [(nid, entry) for nid, entry in reconciliation_map.items() if entry["is_new_topic"]]
    if not new_entries:
        print("Gak ada topik baru yang perlu dilabel Gemini.")
        return reconciliation_map

    print(f"Label {len(new_entries)} topik baru pakai Gemini API...")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    checkpoint = _load_label_checkpoint()
    last_call_time = 0.0

    for i, (new_id, entry) in enumerate(new_entries, start=1):
        final_id = entry["final_topic_id"]
        if final_id in checkpoint:
            entry["topic_label"] = checkpoint[final_id]
            print(f"  [{i}/{len(new_entries)}] Topik {final_id}: skip (dari checkpoint) -> {entry['topic_label']}")
            continue

        elapsed = time.time() - last_call_time
        if elapsed < GEMINI_MIN_INTERVAL_SECONDS:
            time.sleep(GEMINI_MIN_INTERVAL_SECONDS - elapsed)

        top_words = [word for word, _ in new_model.get_topic(new_id)[:10]]
        sample_docs = new_model.get_representative_docs(new_id)[:3]
        prompt = _build_labeling_prompt(top_words, sample_docs)

        label = _call_gemini_with_retry(gemini_model, prompt, final_id)
        last_call_time = time.time()

        entry["topic_label"] = label
        checkpoint[final_id] = label
        _save_label_checkpoint(checkpoint)
        print(f"  [{i}/{len(new_entries)}] Topik {final_id} -> {label}")

    if os.path.exists(GEMINI_CHECKPOINT_PATH):
        os.remove(GEMINI_CHECKPOINT_PATH)
    return reconciliation_map


# ============================================================
# Fungsi 6 -- Bangun dataframe final
# ============================================================

def build_final_dataframe(texts: list[str], topics: np.ndarray, reconciliation_map: dict) -> pd.DataFrame:
    print("Bangun dataframe final (topic_id & label udah dikonsolidasi)...")
    rows = []
    for text, raw_topic_id in zip(texts, topics):
        raw_topic_id = int(raw_topic_id)
        if raw_topic_id == -1:
            rows.append({"utterance": text, "topic_id": -1, "topic_label": "Outlier/Noise", "is_new_topic": False})
            continue
        entry = reconciliation_map[raw_topic_id]
        rows.append({
            "utterance": text,
            "topic_id": entry["final_topic_id"],
            "topic_label": entry["topic_label"],
            "is_new_topic": entry["is_new_topic"],
        })
    df = pd.DataFrame(rows)
    print(f"  -> {len(df)} baris, {df['topic_id'].nunique()} topic_id unik.")
    return df


# ============================================================
# Fungsi 7 -- Hitung metrics (tanpa NMI/ARI/Purity -- no ground truth)
# ============================================================

def compute_metrics(new_model: BERTopic, texts: list[str], topics: np.ndarray, reconciliation_map: dict) -> dict:
    print("Hitung metrics...")
    outlier_ratio = float((topics == -1).sum() / len(topics))
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_new_topics = sum(1 for e in reconciliation_map.values() if e["is_new_topic"])
    n_matched_topics = n_topics - n_new_topics

    rng = random.Random(RANDOM_SEED)
    sample_idx = rng.sample(range(len(texts)), k=min(COHERENCE_SAMPLE_SIZE, len(texts)))
    coherence_texts = [_tokenize_simple(texts[i]) for i in sample_idx]
    dictionary = Dictionary(coherence_texts)

    real_topic_ids = [t for t in set(topics.tolist()) if t != -1]
    topic_word_lists = [[w for w, _ in new_model.get_topic(tid)[:10]] for tid in real_topic_ids]
    topic_word_lists = [words for words in topic_word_lists if words]

    if len(topic_word_lists) >= 2:
        coherence_model = CoherenceModel(
            topics=topic_word_lists, texts=coherence_texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_cv = coherence_model.get_coherence()
    else:
        coherence_cv = float("nan")

    metrics = {
        "outlier_ratio": outlier_ratio,
        "n_topics": n_topics,
        "n_new_topics": n_new_topics,
        "n_matched_topics": n_matched_topics,
        "coherence_cv": coherence_cv,
    }
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    return metrics


# ============================================================
# Fungsi 8 -- Log ke MLflow + register model (TANPA auto-promote)
# ============================================================

def log_to_mlflow(config: dict, new_model: BERTopic, metrics: dict, final_df: pd.DataFrame, is_bootstrap: bool) -> str:
    print("Log run baru ke MLflow (experiment: topic_modeling_production)...")
    mlflow.set_experiment("topic_modeling_production")

    os.makedirs(_TMP_DIR, exist_ok=True)
    today_str = date.today().isoformat()
    csv_filename = f"topic_assignments_{today_str}.csv"
    csv_path = os.path.join(_TMP_DIR, csv_filename)
    final_df.to_csv(csv_path, index=False)

    model_output_dir = os.path.join(_TMP_DIR, MODEL_ARTIFACT_DIR)
    new_model.save(model_output_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=True)

    with mlflow.start_run(run_name=f"retrain_{today_str}") as run:
        run_id = run.info.run_id
        mlflow.log_params(config)
        mlflow.log_param("is_bootstrap", is_bootstrap)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_artifacts(model_output_dir, artifact_path=MODEL_ARTIFACT_DIR)
        mlflow.log_artifact(csv_path)
        print(f"  -> Run '{run.info.run_name}' (run_id={run_id}) selesai di-log.")

        client = MlflowClient()
        try:
            client.create_registered_model(MODEL_REGISTRY_NAME)
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e) and "already exists" not in str(e).lower():
                raise
        # Model disimpan via mlflow.log_artifacts() (format sama seperti
        # modelling.py), BUKAN mlflow.<flavor>.log_model() -- jadi gak ada
        # file MLmodel descriptor. mlflow.register_model() bakal gagal
        # (NOT_FOUND) di kasus ini, makanya pakai create_model_version()
        # langsung, yang menurut dokumentasi resminya memang didesain nerima
        # source runs:/<run_id>/<path> tanpa validasi flavor.
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_DIR}"
        new_version = client.create_model_version(name=MODEL_REGISTRY_NAME, source=model_uri, run_id=run_id)

    print(f"\nModel versi {new_version.version} berhasil di-register.")
    print("REVIEW metrik dulu sebelum promote manual ke Production stage.")
    return run_id


# ============================================================
# main
# ============================================================

def main() -> None:
    print("=" * 70)
    print(f"RETRAIN MODEL -- {date.today().isoformat()}")
    print("=" * 70)

    try:
        print("\nTahap 1/8: Ambil config dari MLflow Model Registry...")
        config, is_bootstrap = get_production_config()

        print("\nTahap 2/8: Ambil data topik dari model Production minggu lalu...")
        old_topics = load_previous_model(is_bootstrap)

        print("\nTahap 3/8: Load data terakumulasi & fit model baru...")
        embeddings, texts = _load_accumulated_data()
        new_model, topics = fit_new_model(embeddings, texts, config)

        print("\nTahap 4/8: Reconcile topik baru vs topik lama...")
        reconciliation_map = reconcile_topics(old_topics, new_model)

        print("\nTahap 5/8: Label topik baru pakai Gemini...")
        reconciliation_map = label_new_topics_with_gemini(new_model, reconciliation_map)

        print("\nTahap 6/8: Bangun dataframe final...")
        final_df = build_final_dataframe(texts, topics, reconciliation_map)

        print("\nTahap 7/8: Hitung metrics...")
        metrics = compute_metrics(new_model, texts, topics, reconciliation_map)

        print("\nTahap 8/8: Log ke MLflow & register model...")
        run_id = log_to_mlflow(config, new_model, metrics, final_df, is_bootstrap)
    except Exception as e:
        print(f"\nGAGAL: {type(e).__name__}: {e}")
        raise

    n_new = sum(1 for e in reconciliation_map.values() if e["is_new_topic"])
    n_matched = len(reconciliation_map) - n_new
    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Total topik   : {len(reconciliation_map)}")
    print(f"Topik baru    : {n_new}")
    print(f"Topik matched : {n_matched}")
    print(f"Run ID        : {run_id}")
    print(f"CSV final     : {os.path.join(_TMP_DIR, f'topic_assignments_{date.today().isoformat()}.csv')} (ter-log sbg artifact)")


if __name__ == "__main__":
    main()
