import json
import shutil
import os

STATE_FILE = "batch_state.json"
QUEUE_DIR = "data_queue"
RAW_DIR = "faq_dataset_raw"

with open(STATE_FILE) as f:
    state = json.load(f)

if state["last_released"] >= state["total_batches"]:
    print("Semua batch sudah dirilis, tidak ada yang baru.")
else:
    next_batch = state["last_released"] + 1
    filename = f"batch_{next_batch:03d}.csv"
    shutil.copy(os.path.join(QUEUE_DIR, filename), os.path.join(RAW_DIR, filename))

    state["last_released"] = next_batch
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Batch {filename} dirilis ke {RAW_DIR}/")
