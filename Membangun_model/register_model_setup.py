"""
register_model_setup.py
Task sekali jalan (one-off, manual dari laptop) buat register model BERTopic
hasil modelling.py ("best_config_single_fit", experiment topic_modeling_basic)
ke MLflow Model Registry, terus promote ke stage "Production".

Tujuannya: retrain_model.py (nanti, di Kriteria 3) bisa ambil config
hyperparameter model Production ini secara otomatis lewat Model Registry,
tanpa perlu hardcode run_id.

Jalankan dari folder Membangun_model/:
    python register_model_setup.py
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"

RUN_ID = "98e51e3766fb4781990d35a9c46daae6"
ARTIFACT_PATH = "bertopic_model_final"
MODEL_NAME = "faq-topic-model"


def setup_mlflow():
    print("[1/4] Setup koneksi MLflow ke DagsHub...")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print(f"  -> Tracking URI di-set ke: {mlflow.get_tracking_uri()}")


def register_model():
    print(f"\n[2/4] Register model dari run '{RUN_ID}' (artifact: '{ARTIFACT_PATH}')...")
    model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
    client = MlflowClient()

    # Catatan: modelling.py nyimpen model pakai mlflow.log_artifacts() (upload
    # file mentah BERTopic), BUKAN mlflow.<flavor>.log_model() -- jadi gak ada
    # file MLmodel descriptor / Logged Model entity di run ini. Fluent API
    # mlflow.register_model() di MLflow 3.x mewajibkan salah satu dari itu,
    # jadi dia bakal gagal (NOT_FOUND). Makanya di sini pakai API level lebih
    # rendah, MlflowClient().create_model_version(), yang menurut dokumentasi
    # resminya memang didesain nerima source runs:/<run_id>/<path> langsung
    # tanpa validasi flavor -- cocok buat model custom kayak BERTopic.
    try:
        client.create_registered_model(MODEL_NAME)
        print(f"  -> Registered model '{MODEL_NAME}' berhasil dibuat.")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            print(f"  -> Registered model '{MODEL_NAME}' sudah ada, lanjut bikin versi baru.")
        else:
            print(f"  -> GAGAL bikin registered model: {type(e).__name__}: {e}")
            raise

    try:
        result = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=RUN_ID)
    except Exception as e:
        print(f"  -> GAGAL register model version: {type(e).__name__}: {e}")
        raise
    print(f"  -> Terdaftar sebagai versi {result.version} (model: '{MODEL_NAME}')")
    return result.version


def promote_to_production(version: str):
    print(f"\n[3/4] Promosikan versi {version} ke stage 'Production'...")
    client = MlflowClient()
    try:
        mv = client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=False,
        )
    except Exception as e:
        print(f"  -> GAGAL promote ke Production: {type(e).__name__}: {e}")
        raise
    print(f"  -> Stage berhasil diubah jadi '{mv.current_stage}' (versi {mv.version})")


def verify_production():
    print("\n[4/4] Verifikasi: cek ulang versi yang aktif di stage 'Production'...")
    client = MlflowClient()
    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception as e:
        print(f"  -> GAGAL verifikasi: {type(e).__name__}: {e}")
        raise

    if not latest:
        raise RuntimeError(
            f"Verifikasi gagal: gak ada versi model '{MODEL_NAME}' di stage 'Production'."
        )

    for mv in latest:
        cocok = "COCOK" if mv.run_id == RUN_ID else "TIDAK COCOK"
        print(f"  -> Versi: {mv.version}, run_id: {mv.run_id} ({cocok} dengan run sumber)")

        run_data = client.get_run(mv.run_id).data
        print(f"\n  Params dari run '{mv.run_id}' (ini yang bakal diambil retrain_model.py):")
        for key, value in run_data.params.items():
            print(f"    {key}: {value}")

    return latest


def main():
    setup_mlflow()
    version = register_model()
    promote_to_production(version)
    latest = verify_production()

    mv = latest[0]
    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    print(f"Nama model : {MODEL_NAME}")
    print(f"Versi      : {mv.version}")
    print(f"Stage      : {mv.current_stage}")
    print(f"Run ID     : {mv.run_id}")
    print(
        f"\nCek juga di https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}, "
        "tab Models — versi ini harusnya sekarang muncul di sana."
    )


if __name__ == "__main__":
    main()
