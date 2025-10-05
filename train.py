import os
from trainer.dataset import fetch_sequences_from_bq
from trainer.model import build_model
import tensorflow as tf

PROJECT = os.getenv("GCP_PROJECT", "your-gcp-project")
BQ_TABLE = os.getenv("BQ_SEQUENCE_TABLE", "project.dataset.sequence_features")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/saved_model")
EPOCHS = int(os.getenv("EPOCHS", "20"))

def main():
    X, y = fetch_sequences_from_bq(PROJECT, BQ_TABLE, max_rows=20000)
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(64).shuffle(1024)

    model = build_model(seq_len=5, feature_dim=9)
    model.fit(ds, epochs=EPOCHS)
    tf.saved_model.save(model, MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
