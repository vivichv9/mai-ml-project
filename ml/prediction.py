import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


class BrokeClassification:
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): features for train model
            target (pd.DataFrame): predicted_value for train model
        """

        X_train, y_train, X_test, y_test = train_test_split(
            data, target, test_size=0.3, random_state=RANDOM_STATE
        )

        self.data = X_train
        self.target = y_train
        self.data_test = X_test
        self.target_test = y_test
        self.model = None

        self.categorical_features = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    def param_selection(self):
        """
        Selection model parameters with Grid search

        Returns:
            Dictionary with best model parameters
        """

        train_data = Pool(
            data=self.data, label=self.target, cat_features=self.categorical_features
        )
        print(train_data.get_features_name())

        param_grid = {
            "iterations": [100, 200, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [1, 2, 3],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [16, 32, 64],
        }

        model = CatBoostClassifier(
            loss_function="MultiClass",
            silent=True,
        )

        grid_search_result = model.grid_search(
            param_grid, train_data, cv=3, verbose=True
        )
        self.model = model
        self.model.set_params(**grid_search_result["params"])

    def train(self):
        self.param_selection()
        self.model.fit(self.data, self.target, cat_features=self.categorical_features)

        test_score = self.model.score(self.data_test, self.target_test)
        print(f"Test score: {test_score}")

    def prediction(self, data):
        prediction = self.model.predict(data=data, prediction_type="Class")
        return prediction


def main():
    br = 1
