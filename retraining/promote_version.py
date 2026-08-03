"""
promote_version.py
Task administratif, dijalankan MANUAL sesekali dari laptop -- promosikan
1 versi model "faq-topic-model" (di MLflow Model Registry, DagsHub) jadi
stage "Production", setelah versi itu direview dan diputuskan layak jadi
baseline resmi (bukan bagian dari GitHub Actions).

Versi lama yang sebelumnya ada di stage Production otomatis ke-archive
(archive_existing_versions=True) -- cuma boleh ada SATU versi aktif di
Production, karena retrain_model.py (get_latest_versions(stages=
["Production"])) mengasumsikan itu buat nemuin baseline yang bener.

Pemakaian (nomor versi lewat command-line argument, BUKAN hardcode):
    python promote_version.py <versi>

Contoh:
    python promote_version.py 3

CATATAN WINDOWS: jalankan dengan $env:PYTHONIOENCODING="utf-8" dulu (PowerShell)
atau set PYTHONIOENCODING=utf-8 (CMD) sebelum run -- MLflow print emoji ke
stdout yang gak kebaca default codepage Windows (cp1252).
"""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

DAGSHUB_USERNAME = "MuhammadHabibna"
DAGSHUB_REPO = "faq-topic-radar"
MODEL_NAME = "faq-topic-model"


def parse_args() -> str:
    parser = argparse.ArgumentParser(
        description="Promosikan 1 versi model faq-topic-model ke stage Production."
    )
    parser.add_argument("version", help="Nomor versi model yang mau dipromote (contoh: 3)")
    return parser.parse_args().version


def setup_mlflow() -> None:
    print("[1/4] Setup koneksi MLflow ke DagsHub...")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print(f"  -> Tracking URI di-set ke: {mlflow.get_tracking_uri()}")


def promote(client: MlflowClient, version: str) -> None:
    print(f"\n[2/4] Cek versi {version} ada di registry '{MODEL_NAME}'...")
    try:
        mv = client.get_model_version(name=MODEL_NAME, version=version)
    except Exception as e:
        print(f"  -> GAGAL: versi {version} gak ketemu di '{MODEL_NAME}': {type(e).__name__}: {e}")
        raise
    print(f"  -> Ketemu. Stage sekarang: '{mv.current_stage}', run_id: {mv.run_id}")

    print(f"\n[3/4] Promosikan versi {version} ke stage 'Production' "
          "(versi lama di Production otomatis di-archive)...")
    try:
        result = client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        print(f"  -> GAGAL promote: {type(e).__name__}: {e}")
        raise
    print(f"  -> Stage berhasil diubah jadi '{result.current_stage}' (versi {result.version})")


def verify(client: MlflowClient, version: str) -> None:
    print("\n[4/4] Verifikasi: cek ulang semua versi & stage-nya...")
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for mv in sorted(all_versions, key=lambda v: int(v.version)):
        marker = " <-- BARU DIPROMOTE" if mv.version == version else ""
        print(f"  Versi {mv.version}: stage='{mv.current_stage}', run_id={mv.run_id}{marker}")

    latest_production = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if len(latest_production) != 1 or latest_production[0].version != version:
        found = [mv.version for mv in latest_production]
        raise RuntimeError(
            f"Verifikasi gagal: stage Production gak nunjuk ke versi {version} "
            f"yang benar (dapat: {found})."
        )
    print(f"\n  -> OK, versi {version} adalah satu-satunya versi di stage Production.")


def main() -> None:
    version = parse_args()

    setup_mlflow()
    client = MlflowClient()
    promote(client, version)
    verify(client, version)

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    print(f"Model : {MODEL_NAME}")
    print(f"Versi : {version}")
    print("Stage : Production")
    print(
        f"\nCek juga di https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}, "
        "tab Models -- versi ini harusnya sekarang muncul sebagai Production."
    )


if __name__ == "__main__":
    main()
