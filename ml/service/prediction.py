import abc
from typing import Dict

import optuna
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 42


class BrokePredictor:
    def __init__(self, data: pd.DataFrame, target: pd.Series, model: CatBoost):
        """
        Args:
            data (pd.DataFrame): Features for training the model
            target (pd.Series): Target variable for training the model
            model (CatBoost): CatBoost model instance (Classifier or Regressor)
        """
        X_train, X_test, y_train, y_test = None, None, None, None
        cat_features = None

        if data is not None and target is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.3, random_state=RANDOM_STATE
            )

            cat_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
        )

        self.data = X_train
        self.target = y_train
        self.data_test = X_test
        self.target_test = y_test
        self.model = model
        self.categorical_features = cat_features
        self.data_val = X_val
        self.target_val = y_val

    def generate_features(
        self, interaction_only=False, degree=2, binning=False, binning_bins=3
    ):
        """
        Generation of additional signals.

        Args:
            interaction_only (bool): If true, then only interacting statements without polynomial ones are created.
            degree (int): The degree of polynomial phenomena.
            binning (bool): If true, binning (categorization) of numeric statements is performed.
            binning_bin (int): a ring of bandages for binning.
        """
        print("Feature generation started...")

        X_train = self.data.copy()
        X_test = self.data_test.copy()

        numerical_features = [
            col
            for col in X_train.select_dtypes(include=["int64", "float64"]).columns
            if col != "index"
        ]
        print(f"Numerical features: {numerical_features}")

        if numerical_features:
            poly = PolynomialFeatures(
                degree=degree, interaction_only=interaction_only, include_bias=False
            )
            poly_features_train = poly.fit_transform(X_train[numerical_features])
            poly_features_test = poly.transform(X_test[numerical_features])

            poly_feature_names = poly.get_feature_names_out(numerical_features)
            poly_df_train = pd.DataFrame(
                poly_features_train, columns=poly_feature_names, index=X_train.index
            )
            poly_df_test = pd.DataFrame(
                poly_features_test, columns=poly_feature_names, index=X_test.index
            )

            poly_df_train = poly_df_train.drop(
                columns=numerical_features, errors="ignore"
            )
            poly_df_test = poly_df_test.drop(
                columns=numerical_features, errors="ignore"
            )

            duplicate_features = set(X_train.columns).intersection(
                set(poly_df_train.columns)
            )
            if duplicate_features:
                raise ValueError(
                    f"Dublicates: {duplicate_features}. Names must be unique!"
                )

            X_train = pd.concat([X_train, poly_df_train], axis=1)
            X_test = pd.concat([X_test, poly_df_test], axis=1)

            print(
                f"Added {poly_features_train.shape[1] - len(numerical_features)} polynomial features."
            )

        if binning:
            for col in numerical_features:
                bin_col = f"{col}_binned"
                try:
                    X_train[bin_col] = pd.qcut(
                        X_train[col], q=binning_bins, duplicates="drop"
                    ).astype(str)
                    X_test[bin_col] = pd.qcut(
                        X_test[col], q=binning_bins, duplicates="drop"
                    ).astype(str)
                    self.categorical_features.append(bin_col)
                    print(f"Feature {col} divided into {binning_bins} bins.")
                except ValueError as e:
                    print(f"Error with binning {col}: {e}")

        categorical_features = self.categorical_features.copy()
        print(f"Categorial features for aggregation: {categorical_features}")

        for col in categorical_features:
            agg_col = f"{col}_count"
            X_train[agg_col] = X_train[col].map(X_train[col].value_counts())
            X_test[agg_col] = X_test[col].map(X_train[col].value_counts())
            print(f"Added aggregating feature {agg_col}.")

        text_features = [
            col
            for col in self.data.columns
            if self.data[col].dtype == "object" and "text" in col.lower()
        ]
        print(f"Text features: {text_features}")

        for col in text_features:
            len_col = f"{col}_length"
            X_train[len_col] = X_train[col].str.len()
            X_test[len_col] = X_test[col].str.len()
            print(f"Added text len feature {len_col}.")

        self.data = X_train
        self.data_test = X_test

        print("Feature generation finished!")

    @abc.abstractmethod
    def param_selection(self):
        """
        Selection model parameters with Optuna
        """
        pass

    def train(self):
        """
        Train model with tuning
        """
        self.param_selection()
        eval_pool = Pool(
            self.data_test, self.target_test, cat_features=self.categorical_features
        )

        self.model.fit(
            self.data,
            self.target,
            cat_features=self.categorical_features,
            eval_set=eval_pool,
            verbose=50,
        )
        self.get_metrics()

    def prediction(self, data):
        """
        Predict data
        """
        prediction = self.model.predict(data=data)
        return prediction

    @abc.abstractmethod
    def save_model(self):
        """
        Save trained model to disk
        """
        pass

    @abc.abstractmethod
    def load_model(self, model_name: str):
        """
        Load trained model from disk
        """
        pass

    @abc.abstractmethod
    def get_metrics(self) -> Dict:
        """
        Evaluate the model and save metrics to a file.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        pass


class BrokeRegressor(BrokePredictor):
    def __init__(self, data, target):
        model = CatBoostRegressor(silent=True)
        super().__init__(data, target, model)

    def param_selection(self):
        if self.model is None:
            raise ValueError("model is None")

        if self.data is None or self.target is None:
            raise ValueError("train data is None")

        train_data = Pool(
            data=self.data, label=self.target, cat_features=self.categorical_features
        )

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 1, 3),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 50),
                "border_count": trial.suggest_int("border_count", 32, 512),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
                "rsm": trial.suggest_float("rsm", 0.1, 1.0),
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["Ordered", "Plain"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
            }

            model = CatBoostRegressor(
                **params,
                cat_features=self.categorical_features,
                random_state=RANDOM_STATE,
                loss_function="MAE",
            )

            eval_pool = Pool(
                self.data_val, self.target_val, cat_features=self.categorical_features
            )
            model.fit(train_data, eval_set=eval_pool, verbose=False)
            preds = model.predict(self.data_test)
            mae = mean_absolute_error(self.target_test, preds)

            return mae

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params

        self.model.set_params(**best_params)

    def save_model(self):
        file_name = "./ml/trained_model/regression.cbm"
        return self.model.save_model(file_name)

    def load_model(self):
        file_name = "./ml/trained_model/regression.cbm"
        return self.model.load_model(file_name)

    def get_metrics(self):
        if self.model is None:
            raise ValueError("model is none")

        if self.data_test is None or self.target_test is None:
            raise ValueError("test data is none")

        preds = self.model.predict(self.data_test)
        metrics = {}

        metrics["MSE"] = mean_squared_error(self.target_test, preds)
        metrics["RMSE"] = mean_squared_error(self.target_test, preds) ** 0.5
        metrics["MAE"] = mean_absolute_error(self.target_test, preds)
        metrics["R2"] = r2_score(self.target_test, preds)
        metrics["Explained Variance"] = explained_variance_score(
            self.target_test, preds
        )

        filename = "./metrics/regression_metrics.txt"

        with open(filename, "w") as file:
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    file.write(f"{metric}:\n")
                    for key, val in value.items():
                        file.write(f"  {key}: {val}\n")
                else:
                    file.write(f"{metric}: {value}\n")

        print(f"\n--- Метрики модели: Классификация ---")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for key, val in value.items():
                    print(f"  {key}: {val}")
            else:
                print(f"{metric}: {value}")
        print("--- Конец метрик ---\n")

        return metrics


class BrokeClassifier(BrokePredictor):
    def __init__(self, data, target):
        model = CatBoostClassifier(silent=True)
        super().__init__(data, target, model)

    def param_selection(self):
        if self.model is None:
            raise ValueError("model is None")

        if self.data is None or self.target is None:
            raise ValueError("train data is None")

        train_data = Pool(
            data=self.data, label=self.target, cat_features=self.categorical_features
        )

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 1, 3),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 50),
                "border_count": trial.suggest_int("border_count", 32, 512),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
                "rsm": trial.suggest_float("rsm", 0.1, 1.0),
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["Ordered", "Plain"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
            }

            params["eval_metric"] = "MultiClass"
            model = CatBoostClassifier(
                **params,
                cat_features=self.categorical_features,
                random_state=RANDOM_STATE,
            )

            eval_pool = Pool(
                self.data_val, self.target_val, cat_features=self.categorical_features
            )
            model.fit(
                train_data, eval_set=eval_pool, early_stopping_rounds=50, verbose=False
            )
            preds = model.predict(self.data_test)
            score = recall_score(self.target_test, preds, average="macro")
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params

        self.model.set_params(**best_params)

    def save_model(self):
        file_name = "./ml/trained_model/classification.cbm"
        return self.model.save_model(file_name)

    def load_model(self):
        file_name = "./ml/trained_model/classification.cbm"
        return self.model.load_model(file_name)

    def get_metrics(self):
        if self.model is None:
            raise ValueError("model is none")

        if self.data_test is None or self.target_test is None:
            raise ValueError("test data is none")

        preds = self.model.predict(self.data_test)
        preds_proba = self.model.predict_proba(self.data_test)

        metrics = {}
        metrics["RocAuc Score"] = roc_auc_score(
            self.target_test, preds_proba, multi_class="ovo"
        )
        metrics["Accuracy"] = accuracy_score(self.target_test, preds)
        metrics["Recall"] = recall_score(self.target_test, preds, average="macro")
        metrics["Precision"] = precision_score(self.target_test, preds, average="macro")
        metrics["F1 Score"] = f1_score(self.target_test, preds, average="macro")
        metrics["Confusion Matrix"] = confusion_matrix(self.target_test, preds).tolist()
        metrics["Classification Report"] = classification_report(
            self.target_test, preds, output_dict=True
        )

        filename = "./metrics/classification_metrics.txt"

        with open(filename, "w") as file:
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    file.write(f"{metric}:\n")
                    for key, val in value.items():
                        file.write(f"  {key}: {val}\n")
                else:
                    file.write(f"{metric}: {value}\n")

        print(f"\n--- Метрики модели ('Регрессия') ---")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for key, val in value.items():
                    print(f"  {key}: {val}")
            else:
                print(f"{metric}: {value}")
        print("--- Конец метрик ---\n")

        return metrics
