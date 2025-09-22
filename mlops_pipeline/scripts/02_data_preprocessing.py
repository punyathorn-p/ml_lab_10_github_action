import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mlflow


def preprocess_data(test_size=0.25, random_state=42):
    """
    Loads raw data, splits it into training and testing sets,
    and logs the resulting datasets as artifacts in MLflow.
    """
    mlflow.set_experiment("Breast Cancer - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # 1. Load data as a DataFrame
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame

        # 2. Split the data into training and testing sets
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. Create a temporary directory to save processed data
        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)

        # Recombine features and target for easy saving
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)
        print(f"Saved processed data to '{processed_data_dir}' directory.")

        # 4. Log parameters and metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))

        # 5. Log the processed data directory as an artifact
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")
        
        print("-" * 50)
        print(f"Data preprocessing run finished. Preprocessing Run ID:")
        print(f"{run_id}")
        print("-" * 50)


if __name__ == "__main__":
    preprocess_data()
