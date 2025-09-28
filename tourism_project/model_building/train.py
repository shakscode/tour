# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# =============================
# MLflow setup
# =============================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("shaks_tourism_experiment")

# =============================
# Hugging Face setup
# =============================
api = HfApi(token=os.getenv("HF_TOKEN"))

# =============================
# Download dataset from Hugging Face
# =============================
repo_id = "ShaksML/tourism"

Xtrain_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # ensure it's a Series
ytest = pd.read_csv(ytest_path).squeeze()

# =============================
# Features
# =============================
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# =============================
# Class imbalance handling
# =============================
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__colsample_bytree": [0.4, 0.5],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5],
    "xgbclassifier__learning_rate": [0.01, 0.05],
    "xgbclassifier__reg_lambda": [0.4, 0.5],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# =============================
# Training with MLflow logging
# =============================
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log only the best model params
    mlflow.log_params(grid_search.best_params_)

    # Best model
    top_model = grid_search.best_estimator_
    classification_threshold = 0.45

    # Predictions
    y_pred_train = (top_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (top_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    # Save the model locally
    model_path = "top_tourism_model_v1.joblib"
    joblib.dump(top_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # =============================
    # Upload to Hugging Face
    # =============================
    model_repo_id = repo_id  # reuse same repo for models
    repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Repo '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{model_repo_id}' not found. Creating new repo...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
