import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature

def train_evaluate_register(preprocessing_run_id, C=1.0):
    ACCURACY_THRESHOLD = 0.95

    # กำหนด MLflow experiment และ artifact location ภายใน project
    mlruns_dir = os.path.abspath("mlruns")
    exp = mlflow.set_experiment(
        experiment_name="Breast Cancer - Model Training",
        artifact_location=f"file://{mlruns_dir}"
    )
    print(f"MLflow experiment artifact location: {exp.artifact_location}")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # 1. Load preprocessed data from artifacts
        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")

            train_path = os.path.join(local_artifact_path, "train.csv")
            test_path = os.path.join(local_artifact_path, "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            sys.exit(1)

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # 2. Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)

        # 3. Evaluate model
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # 4. Log parameters, metrics, and model with signature and input_example
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.iloc[:5]

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="cancer_classifier_pipeline",
            signature=signature,
            input_example=input_example
        )
        print("Model logged successfully.")

        # 5. Register model if accuracy meets threshold
        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets the threshold. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/cancer_classifier_pipeline"
            registered_model = mlflow.register_model(model_uri, "breast-cancer-classifier-prod")
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")

        print("Training run finished.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)

    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
