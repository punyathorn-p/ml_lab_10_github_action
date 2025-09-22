import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts

# ล้างค่า environment variable เก่า
for var in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_URI"]:
    if var in os.environ:
        del os.environ[var]

# ตั้งค่า MLflow ให้เก็บ artifact ภายใน project
mlruns_dir = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
print(f"MLflow tracking URI set to: file://{mlruns_dir}")

mlflow.set_experiment("Breast Cancer - Model Training")

def train_evaluate_register(preprocessing_run_id, C=1.0):
    ACCURACY_THRESHOLD = 0.95

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
        mlflow.log_param("C", C)

        # โหลด artifact จาก preprocessing
        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")

            train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
            test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))
            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            sys.exit(1)

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # สร้าง pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)

        # ประเมินผล
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        # log model แบบใช้ name แทน artifact_path
        mlflow.sklearn.log_model(pipeline, artifact_path=None, registered_model_name="breast-cancer-classifier-prod", name="cancer_classifier_pipeline")

        # ตรวจสอบ threshold
        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets threshold, registered automatically.")
        else:
            print(f"Accuracy below threshold ({acc:.4f}), not registering.")
        print("Training run finished.")


if __name__ == "__main__":
    if not os.path.exists("preprocessing_run_id.txt"):
        print("Error: preprocessing_run_id.txt not found!")
        sys.exit(1)

    with open("preprocessing_run_id.txt") as f:
        run_id = f.read().strip()

    c_value = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
