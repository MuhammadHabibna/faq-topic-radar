"""
backfill_v1_labels.py
Task sekali jalan (one-off, manual dari laptop) buat melabeli topik dari
model yang LAGI berstatus Production ("faq-topic-model") yang gak pernah
punya artifact topic_assignments_*.csv -- karena run itu hasil modelling.py
(Kriteria 2), yang gak menghasilkan CSV label per-topik kayak retrain_model.py.

Cara kerja: download folder model ("bertopic_model_final") lengkap dari run
Production, load modelnya, ambil top-10 kata c-TF-IDF tiap topik langsung
dari model (TANPA re-embed data mentah / tanpa sample docs), kirim ke Gemini
buat dikasih 1 label singkat, simpan sebagai topic_labels_v1_baseline.json,
lalu upload balik ke run yang SAMA (bukan run baru) sebagai artifact baru.

Setelah ini, retrain_model.py bisa nemuin label v1 lewat artifact baru ini
di load_previous_model() (branch "csv_path is None").

Autentikasi MLflow ke DagsHub: mlflow.set_tracking_uri() + env var
MLFLOW_TRACKING_USERNAME/PASSWORD (BUKAN dagshub.init() -- pernah memicu
OAuth interaktif yang gagal di lingkungan headless).

Jalankan dari folder Membangun_model/ (dependency bertopic/mlflow/
google-generativeai sudah ada dari task sebelumnya):
    python backfill_v1_labels.py

CATATAN WINDOWS: jalankan dengan $env:PYTHONIOENCODING="utf-8" dulu (PowerShell)
atau set PYTHONIOENCODING=utf-8 (CMD) sebelum run -- MLflow print emoji ke
stdout yang gak kebaca default codepage Windows (cp1252).
"""

import json
import os
import time

import google.generativeai as genai
import mlflow
from bertopic import BERTopic
from google.api_core.exceptions import ResourceExhausted
from mlflow.tracking import MlflowClient

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"
MODEL_REGISTRY_NAME = "faq-topic-model"
MODEL_ARTIFACT_DIR = "bertopic_model_final"

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_MIN_INTERVAL_SECONDS = 4.5
GEMINI_MAX_RETRIES = 5
GEMINI_CHECKPOINT_PATH = "backfill_label_checkpoint.json"

OUTPUT_JSON_PATH = "topic_labels_v1_baseline.json"


def _connect_mlflow() -> None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")


def get_production_run_id(client: MlflowClient) -> str:
    print(f"[1/5] Cari versi '{MODEL_REGISTRY_NAME}' yang lagi di stage Production...")
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"Gak ada versi '{MODEL_REGISTRY_NAME}' di stage Production -- gak ada yang bisa dibackfill."
        )
    production_version = versions[0]
    print(f"  -> Versi {production_version.version}, run_id={production_version.run_id}")
    return production_version.run_id


def download_and_load_model(run_id: str) -> BERTopic:
    print(f"\n[2/5] Download folder artifact '{MODEL_ARTIFACT_DIR}' dari run {run_id}...")
    model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=MODEL_ARTIFACT_DIR)
    print(f"  -> Ter-download ke: {model_dir}")
    print("  -> Load model BERTopic...")
    topic_model = BERTopic.load(model_dir)
    print("  -> Model ter-load.")
    return topic_model


def _load_checkpoint() -> dict[int, str]:
    if os.path.exists(GEMINI_CHECKPOINT_PATH):
        with open(GEMINI_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  -> Checkpoint lama ketemu: {len(data)} topik udah pernah kelabel, di-skip.")
        return {int(k): v for k, v in data.items()}
    return {}


def _save_checkpoint(checkpoint: dict[int, str]) -> None:
    with open(GEMINI_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


def _build_labeling_prompt(top_words: list[str]) -> str:
    words_str = ", ".join(top_words)
    return f"""You are a topic modeling expert. Based on the top keywords below, give ONE
short, human-readable label for this topic (example: "Payment & Billing Issues").

Top keywords (c-TF-IDF): {words_str}

Answer with ONLY the short label (max 5 words). The label MUST be in English,
regardless of the language of the keywords above. No extra explanation."""


def _is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, ResourceExhausted):
        return True
    message = str(e).lower()
    return "429" in message or "quota" in message or "rate limit" in message


def _call_gemini_with_retry(gemini_model, prompt: str, topic_id: int) -> str:
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if not _is_rate_limit_error(e):
                print(f"    GAGAL labeling topik {topic_id} ({type(e).__name__}: {e})")
                raise
            wait = GEMINI_MIN_INTERVAL_SECONDS * (2 ** (attempt - 1))
            print(f"    Rate limit (429) buat topik {topic_id}, percobaan {attempt}/{GEMINI_MAX_RETRIES}. "
                  f"Nunggu {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Gagal labeling topik {topic_id} setelah {GEMINI_MAX_RETRIES}x percobaan (rate limit terus).")


def label_all_topics(topic_model: BERTopic) -> dict[int, str]:
    topic_ids = sorted(t for t in topic_model.get_topic_info()["Topic"].tolist() if t != -1)
    print(f"\n[3/5] Label {len(topic_ids)} topik pakai Gemini API (top-10 kata c-TF-IDF aja, tanpa sample docs)...")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    checkpoint = _load_checkpoint()
    last_call_time = 0.0
    labels: dict[int, str] = {}

    for i, topic_id in enumerate(topic_ids, start=1):
        if topic_id in checkpoint:
            labels[topic_id] = checkpoint[topic_id]
            print(f"  [{i}/{len(topic_ids)}] Topik {topic_id}: skip (dari checkpoint) -> {labels[topic_id]}")
            continue

        elapsed = time.time() - last_call_time
        if elapsed < GEMINI_MIN_INTERVAL_SECONDS:
            time.sleep(GEMINI_MIN_INTERVAL_SECONDS - elapsed)

        top_words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        prompt = _build_labeling_prompt(top_words)

        label = _call_gemini_with_retry(gemini_model, prompt, topic_id)
        last_call_time = time.time()

        labels[topic_id] = label
        checkpoint[topic_id] = label
        _save_checkpoint(checkpoint)
        print(f"  [{i}/{len(topic_ids)}] Topik {topic_id} -> {label}")

    if os.path.exists(GEMINI_CHECKPOINT_PATH):
        os.remove(GEMINI_CHECKPOINT_PATH)
    return labels


def save_and_upload(labels: dict[int, str], client: MlflowClient, run_id: str) -> None:
    print(f"\n[4/5] Simpan hasil ke '{OUTPUT_JSON_PATH}'...")
    labels_str_keys = {str(k): v for k, v in labels.items()}
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_str_keys, f, indent=2, ensure_ascii=False)
    print(f"  -> {len(labels)} label tersimpan lokal.")

    print(f"\n[5/5] Upload '{OUTPUT_JSON_PATH}' sebagai artifact baru ke run {run_id}...")
    client.log_artifact(run_id, local_path=OUTPUT_JSON_PATH)
    print("  -> Upload selesai.")


def main() -> None:
    print("=" * 70)
    print("BACKFILL LABEL v1 -- one-off")
    print("=" * 70)

    _connect_mlflow()
    client = MlflowClient()

    run_id = get_production_run_id(client)
    topic_model = download_and_load_model(run_id)
    labels = label_all_topics(topic_model)
    save_and_upload(labels, client, run_id)

    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Topik berhasil dilabeli : {len(labels)}")
    print(f"Artifact baru           : {OUTPUT_JSON_PATH} (run {run_id})")
    print(
        "\nCek di MLflow UI run v1, tab Artifacts -- harusnya sekarang ada "
        f"file {OUTPUT_JSON_PATH} di root (sejajar folder {MODEL_ARTIFACT_DIR}/)."
    )


if __name__ == "__main__":
    main()
