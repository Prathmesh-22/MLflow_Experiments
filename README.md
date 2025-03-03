MLFLOW EXPERIMENTS BREAKDOWN


## **1️⃣ Import Required Libraries**
```python
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging
```
- **os, warnings, sys** → System-related modules (handling warnings, arguments, etc.).
- **pandas (pd)** → Used for handling tabular data.
- **numpy (np)** → Used for numerical operations.
- **sklearn.metrics** → Functions to evaluate model performance:
  - `mean_squared_error`, `mean_absolute_error`, `r2_score`
- **sklearn.model_selection** → `train_test_split` for splitting data.
- **sklearn.linear_model** → `ElasticNet` regression model (combines Lasso & Ridge).
- **urllib.parse.urlparse** → Parses URLs.
- **mlflow** → Library for tracking experiments, logging parameters & models.
- **mlflow.sklearn** → MLflow functions specific to scikit-learn models.
- **logging** → Helps in logging errors/warnings.

---

## **2️⃣ Logging Setup**
```python
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```
- Configures logging to **show warnings or errors**.
- Creates a logger named `__name__` (script name).

---

## **3️⃣ Define Evaluation Metrics**
```python
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))  # Root Mean Squared Error
    mae = mean_absolute_error(actual, pred)  # Mean Absolute Error
    r2 = r2_score(actual, pred)  # R² Score
    return rmse, mae, r2
```
- This function calculates:
  - **RMSE**: Measures average model error (lower is better).
  - **MAE**: Measures absolute differences (lower is better).
  - **R²**: Measures how well the model explains the variance (closer to 1 is better).

---

## **4️⃣ Main Program Execution**
```python
if __name__ == "__main__":
```
- Ensures this code **only runs** when the script is executed directly (not imported).

---

## **5️⃣ Ignore Warnings & Set Seed**
```python
    warnings.filterwarnings("ignore")  # Ignores warnings for cleaner output
    np.random.seed(40)  # Ensures reproducibility of random operations
```

---

## **6️⃣ Load Dataset**
```python
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
```
- Loads the **wine quality dataset** from a remote URL.
- Uses **try-except** to catch errors (e.g., no internet).
- Data is stored in a **Pandas DataFrame**.

---

## **7️⃣ Split Data into Training & Test Sets**
```python
    train, test = train_test_split(data)
```
- **Splits dataset** randomly into:
  - **75% training data**
  - **25% testing data** (default behavior of `train_test_split`).

---

## **8️⃣ Define Features & Target**
```python
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
```
- **Features (`X`)**: All columns **except** `"quality"`.
- **Target (`y`)**: `"quality"` column (what we're predicting).

---

## **9️⃣ Read Command-line Arguments**
```python
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
```
- Reads **command-line arguments** for `alpha` and `l1_ratio`.
- If no arguments are provided, **default values** are:
  - `alpha = 0.5`
  - `l1_ratio = 0.5`

---

## **🔟 Train the ElasticNet Model & Track in MLflow**
```python
    with mlflow.start_run():
```
- **Starts an MLflow experiment run**.

```python
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
```
- Creates an **ElasticNet** model using `alpha` & `l1_ratio`.
- Trains it on the training dataset.

```python
        predicted_qualities = lr.predict(test_x)
```
- Predicts wine quality on the **test set**.

---

## **1️⃣1️⃣ Evaluate Model Performance**
```python
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
```
- Calls `eval_metrics()` to calculate RMSE, MAE, and R².

```python
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
```
- Prints **model performance metrics**.

---

## **1️⃣2️⃣ Log Parameters & Metrics in MLflow**
```python
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
```
- Logs **hyperparameters** in MLflow.

```python
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
```
- Logs **performance metrics**.

---

## **1️⃣3️⃣ Register Model in MLflow**
```python
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)
```
- Generates model **signature** (helps in deployment).

---

### **1️⃣4️⃣ DAGShub Integration (Optional)**
```python
        ## For Remote server only (DAGShub)

        # remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)
```
- **DAGShub** is a remote MLflow tracking server.
- Commented out in this script.

---

### **1️⃣5️⃣ Log & Register the Model**
```python
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
```
- Retrieves **tracking URL** type.

```python
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
```
- If **MLflow is running on a remote server**, it **registers the model**.
- Otherwise, it just saves it locally.

---

## **✅ Summary**
### **What this script does:**
1. Loads the **Wine Quality dataset** from a URL.
2. Splits the data into **train (75%) & test (25%)**.
3. Defines **features (`X`)** and **target (`y`)**.
4. Reads **command-line arguments** for `alpha` & `l1_ratio`.
5. Trains an **ElasticNet Regression** model.
6. **Evaluates performance** (RMSE, MAE, R²).
7. **Logs parameters & metrics** in MLflow.
8. **Saves & registers the model** (if remote MLflow is set up).

This is a **basic MLOps workflow** where MLflow is used for experiment tracking. 🚀 Let me know if you have any doubts! 😊
