import os
import shutil
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# -----------------------------
# ล้าง cache MLflow เก่า
# -----------------------------
shutil.rmtree(os.path.expanduser("~/.mlflow"), ignore_errors=True)
for var in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_URI"]:
    if var in os.environ:
        del os.environ[var]

import mlflow

# -----------------------------
# ตั้งค่า MLflow ให้ใช้ folder ภายใน project
# -----------------------------
mlruns_dir = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
print(f"MLflow tracking URI set to: file://{mlruns_dir}")

# ใช้ experiment ใหม่
mlflow.set_experiment("Breast Cancer - Data Preprocessing New")

def preprocess_data(test_size=0.25, random_state=42):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = load_breast_cancer(as_frame=True).frame
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        processed_data_dir = os.path.abspath("processed_data")
        os.makedirs(processed_data_dir, exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)
        print(f"Saved processed data to '{processed_data_dir}' directory.")

        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")

        # บันทึก run_id สำหรับ GitHub Actions
        run_id_file = "mlops_pipeline/preprocessing_run_id.txt"
        os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
        with open(run_id_file, "w") as f:
            f.write(run_id)
        print(f"Saved preprocessing run_id to '{run_id_file}'")

        print("-" * 50)
        print(f"Data preprocessing run finished. Preprocessing Run ID: {run_id}")
        print("-" * 50)

        return run_id

if __name__ == "__main__":
    preprocess_data()
