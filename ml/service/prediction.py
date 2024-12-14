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

        X_train = None
        y_train = None
        X_test = None
        y_test = None
        cat_features = None

        if data is not None and target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.3, random_state=RANDOM_STATE
            )

            cat_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self.data = X_train
        self.target = y_train
        self.data_test = X_test
        self.target_test = y_test
        self.model = None
        self.categorical_features = cat_features

    def param_selection(self):
        """
        Selection model parameters with Grid search

        Returns:
            Dictionary with best model parameters
        """

        train_data = Pool(
            data=self.data, label=self.target, cat_features=self.categorical_features
        )
        # print(train_data.get_features_name())

        param_grid = {
            "iterations": [100],
            "learning_rate": [0.05],
            "depth": [1, 2],
            "l2_leaf_reg": [1, 3],
            "border_count": [16],
        }

        model = CatBoostClassifier(
            loss_function="MultiClass",
            silent=False,
        )

        grid_search_result = model.grid_search(
            param_grid, train_data, cv=3, verbose=True
        )
        self.model = model

    def train(self):
        self.model.fit(self.data, self.target, cat_features=self.categorical_features)

        test_score = self.model.score(self.data_test, self.target_test)
        print(f"Test score: {test_score}")

    def prediction(self, data):
        prediction = self.model.predict(data=data, prediction_type="Class")
        return prediction

    def save_model(self):
        self.model.save_model("./ml/trained_model/classification.cbm", format="cbm")

    def load_model(self):
        self.model = CatBoostClassifier()
        self.model.load_model("./ml/trained_model/classification.cbm")
