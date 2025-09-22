from sklearn.datasets import load_breast_cancer
import mlflow

def load_and_predict():
    """
    Simulates a production scenario by loading a model from a specific
    stage in the MLflow Model Registry and using it for prediction.
    """
    # Set model name and stage here
    MODEL_NAME = "breast-cancer-classifier-prod"
    MODEL_STAGE = "Staging"  # สามารถเปลี่ยนเป็น "Production" ได้เมื่อพร้อม

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    # Load the model from the Model Registry
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    # Prepare new sample data for prediction (using the first row of the dataset)
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    sample_data = X[0:1]  # Using the first row as a sample
    actual_label = y[0]

    # Use the loaded model to make a prediction
    # No manual preprocessing is needed because we logged the entire pipeline
    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)

if __name__ == "__main__":
    load_and_predict()
