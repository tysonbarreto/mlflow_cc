# Load the model back for predictions as a generic Python Function model
import mlflow
from mlflow.models import infer_signature, validate_serving_input

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow import MlflowClient

client = MlflowClient()

mlflow.set_tracking_uri(uri='http://127.0.0.1:5000')

logged_model = 'runs:/39562edd925d4a0bb264fb156c482203/iris_model'
model_name = "tracking-quickstart"
model_version = "3"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


# Predict on a Pandas DataFrame.
import pandas as pd

prediction = loaded_model.predict(pd.DataFrame(X_test,
                                  columns=datasets.load_iris().feature_names)
                     )

print(prediction)
print(datasets.load_iris().feature_names)

latest = client.get_latest_versions("tracking-quickstart",stages=["None"])
print(latest)