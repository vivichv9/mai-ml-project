import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


class BrokePredictor:
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame, model: CatBoost):
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
        self.model = model
        self.categorical_features = cat_features

    def param_selection(self):
        """
        Selection model parameters with Optuna

        Returns:
            Dictionary with best model parameters
        """
        if self.model is None:
            raise ValueError("model is None")

        if self.data is None or self.target is None:
            raise ValueError("train data is None")

        train_data = Pool(
            data=self.data, label=self.target, cat_features=self.categorical_features
        )

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 1, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 16, 255),
                "random_seed": RANDOM_STATE,
            }

            model = self.model.set_params(**params)

            model.fit(train_data, verbose=0)
            preds = model.predict(self.data_test)
            model_type = type(self.model)

            if model_type is CatBoostRegressor:
                return mean_squared_error(self.target_test, preds, squared=False)

            return recall_score(self.target_test, preds, average="macro")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_seed"] = RANDOM_STATE

        self.model = self.model.set_params(**best_params)


    def train(self):
        self.model.fit(self.data, self.target, cat_features=self.categorical_features)

        test_score = self.model.score(self.data_test, self.target_test)
        print(f"Test score: {test_score}")

    def prediction(self, data):
        prediction = self.model.predict(data=data, prediction_type="Class")
        return prediction

    def save_model(self, model_name: str):
        self.model.save_model(f"./ml/trained_model/{model_name}.cbm", format="cbm")

    def load_model(self, model_name: str):
        self.model = CatBoostClassifier()
        self.model.load_model(f"./ml/trained_model/{model_name}.cbm")
