
import time

import mlflow
from mlflow import MlflowClient


# --------------------------------------------------
# 1. Connect to our MLflow server
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()


# --------------------------------------------------
# 2. Define our registered model name
# --------------------------------------------------

MODEL_NAME = "CustomerChurn"


# --------------------------------------------------
# 3. Identify our two trained models
# --------------------------------------------------

models = {
    "random_forest": {
        "run_id": "f8ffcaad468d43f2b2b0462d47c1caa2",
        "alias": "champion"
    },

    "gradient_boosting": {
        "run_id": "85863358f74b4a7fac4c8fa2fb4d152f",
        "alias": "challenger"
    }
}


# --------------------------------------------------
# 4. Register each model in MLflow
# --------------------------------------------------

for model_name, info in models.items():

    print(f"\nRegistering {model_name}...")

    run_id = info["run_id"]
    alias = info["alias"]

    # Find the model saved inside the training run
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    version = registered_model.version

    print(f"Created version {version} for {model_name}")

    # Wait until MLflow finishes saving the model
    for attempt in range(60):

        model_version = client.get_model_version(
            name=MODEL_NAME,
            version=version
        )

        if model_version.status == "READY":
            break

        if model_version.status == "FAILED":
            raise RuntimeError(
                f"Registration failed for {model_name}"
            )

        time.sleep(1)

    else:
        raise TimeoutError(
            f"Registration timed out for {model_name}"
        )

    # Give the model its alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version
    )

    print(
        f"{model_name} successfully registered "
        f"as version {version} with alias @{alias}"
    )


# --------------------------------------------------
# 5. Verify our aliases
# --------------------------------------------------

print("\nChecking registered models...")

for alias in ["champion", "challenger"]:

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=alias
    )

    print(
        f"@{alias} points to version "
        f"{model_version.version}"
    )


print("\nModel Registry setup completed!")
