import optuna
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_squared_error,
    recall_score,
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import os
from feature_engine.encoding import MeanEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np

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

        self.data = X_train
        self.target = y_train
        self.data_test = X_test
        self.target_test = y_test
        self.model = model
        self.categorical_features = cat_features

    def generate_features(self, interaction_only=False, degree=2, binning=False, binning_bins=3):
        """
        Генерация дополнительных признаков.

        Args:
            interaction_only (bool): Если True, то создаются только взаимодействующие признаки без полиномиальных.
            degree (int): Степень полиномиальных признаков.
            binning (bool): Если True, выполняется биннинг (разбиение на категории) числовых признаков.
            binning_bins (int): Количество бинов для биннинга.
        """
        print("Начинаем генерацию дополнительных признаков...")

        X_train = self.data.copy()
        X_test = self.data_test.copy()

        numerical_features = [col for col in X_train.select_dtypes(include=['int64', 'float64']).columns if col != 'index']
        print(f"Числовые признаки для полиномиальных признаков: {numerical_features}")

        if numerical_features:
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
            poly_features_train = poly.fit_transform(X_train[numerical_features])
            poly_features_test = poly.transform(X_test[numerical_features])

            poly_feature_names = poly.get_feature_names_out(numerical_features)
            poly_df_train = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=X_train.index)
            poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=X_test.index)

            poly_df_train = poly_df_train.drop(columns=numerical_features, errors='ignore')
            poly_df_test = poly_df_test.drop(columns=numerical_features, errors='ignore')

            duplicate_features = set(X_train.columns).intersection(set(poly_df_train.columns))
            if duplicate_features:
                raise ValueError(f"Дублирующиеся имена признаков: {duplicate_features}. Убедитесь, что все имена уникальны.")

            X_train = pd.concat([X_train, poly_df_train], axis=1)
            X_test = pd.concat([X_test, poly_df_test], axis=1)

            print(f"Добавлено {poly_features_train.shape[1] - len(numerical_features)} полиномиальных признаков.")

        if binning:
            for col in numerical_features:
                bin_col = f"{col}_binned"
                try:
                    X_train[bin_col] = pd.qcut(X_train[col], q=binning_bins, duplicates='drop').astype(str)
                    X_test[bin_col] = pd.qcut(X_test[col], q=binning_bins, duplicates='drop').astype(str)
                    self.categorical_features.append(bin_col)
                    print(f"Признак {col} разбит на {binning_bins} бинов.")
                except ValueError as e:
                    print(f"Ошибка при биннинге признака {col}: {e}")

        categorical_features = self.categorical_features.copy() 
        print(f"Категориальные признаки для агрегации: {categorical_features}")

        for col in categorical_features:
            agg_col = f"{col}_count"
            X_train[agg_col] = X_train[col].map(X_train[col].value_counts())
            X_test[agg_col] = X_test[col].map(X_train[col].value_counts()) 
            print(f"Добавлен агрегированный признак {agg_col}.")

        text_features = [col for col in self.data.columns if self.data[col].dtype == 'object' and 'text' in col.lower()]
        print(f"Текстовые признаки для обработки: {text_features}")

        for col in text_features:
            len_col = f"{col}_length"
            X_train[len_col] = X_train[col].str.len()
            X_test[len_col] = X_test[col].str.len()
            print(f"Добавлен признак длины текста {len_col}.")

        self.data = X_train
        self.data_test = X_test

        print("Генерация дополнительных признаков завершена.")

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
                "depth": trial.suggest_int("depth", 1, 3),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 50),
                "border_count": trial.suggest_int("border_count", 32, 512),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
                "rsm": trial.suggest_float("rsm", 0.1, 1.0),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
            }

            model_type = type(self.model)
            model = None

            if model_type is CatBoostRegressor:
                model = CatBoostRegressor(**params, cat_features=self.categorical_features, random_state=RANDOM_STATE, loss_function='MAE')

            elif model_type is CatBoostClassifier:
                params["eval_metric"] = "AUC"
                model = CatBoostClassifier(**params, cat_features=self.categorical_features, random_state=RANDOM_STATE)

            else:   
                raise ValueError("Unsupported model type for parameter selection")

            if model_type is CatBoostClassifier:
                eval_pool = Pool(self.data_test, self.target_test, cat_features=self.categorical_features)
                model.fit(train_data, eval_set=eval_pool, early_stopping_rounds=50, verbose=False)
                preds = model.predict_proba(self.data_test)
                score = roc_auc_score(self.target_test, preds, multi_class="ovo", average="macro")
                return score

            else:
                model.fit(train_data, verbose=False)
                preds = model.predict(self.data_test)
                mae = mean_absolute_error(self.target_test, preds)
                return mae

        model_type = type(self.model)
        study = None
        if model_type is CatBoostRegressor:
            study = optuna.create_study(direction="minimize")
        elif model_type is CatBoostClassifier:
            study = optuna.create_study(direction="maximize")
        else:
            raise ValueError("Unsupported model type for Optuna study")

        study.optimize(objective, n_trials=10)
        best_params = study.best_params

        print(f"Лучшие параметры: {best_params}")

        self.model.set_params(**best_params)

    def train(self):
        self.param_selection()
        eval_pool = Pool(self.data_test, self.target_test, cat_features=self.categorical_features)
        self.model.fit(
            self.data,
            self.target,
            cat_features=self.categorical_features,
            eval_set=eval_pool,
            verbose=50
        )
        self.get_metrics()

    def prediction(self, data):
        prediction = self.model.predict(data=data)
        return prediction

    def save_model(self, model_name: str):
        self.model.save_model(f"./ml/trained_model/{model_name}.cbm", format="cbm")

    def load_model(self, model_name: str):
        if isinstance(self.model, CatBoostClassifier):
            self.model = CatBoostClassifier()
        elif isinstance(self.model, CatBoostRegressor):
            self.model = CatBoostRegressor()
        else:
            raise ValueError("Unsupported model type for loading")
        self.model.load_model(f"./ml/trained_model/{model_name}.cbm")

    def get_metrics(self):
        """
        Evaluate the model and save metrics to a file.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        if self.model is None:
            raise ValueError("model is none")

        if self.data_test is None or self.target_test is None:
            raise ValueError("test data is none")

        preds = self.model.predict(self.data_test)
        metrics = {}

        if isinstance(self.model, CatBoostRegressor):
            metrics["MSE"] = mean_squared_error(self.target_test, preds)
            metrics["RMSE"] = mean_squared_error(self.target_test, preds) ** 0.5
            metrics["MAE"] = mean_absolute_error(self.target_test, preds)
            metrics["R2"] = r2_score(self.target_test, preds)
            metrics["Explained Variance"] = explained_variance_score(
                self.target_test, preds
            )

        elif isinstance(self.model, CatBoostClassifier):
            metrics["Accuracy"] = accuracy_score(self.target_test, preds)
            metrics["Recall"] = recall_score(self.target_test, preds, average="macro")
            metrics["Precision"] = precision_score(
                self.target_test, preds, average="macro"
            )
            metrics["F1 Score"] = f1_score(self.target_test, preds, average="macro")
            metrics["Confusion Matrix"] = confusion_matrix(
                self.target_test, preds
            ).tolist()
            metrics["Classification Report"] = classification_report(
                self.target_test, preds, output_dict=True
            )

        else:
            raise ValueError("Unsupported model type")

        model_type = type(self.model)
        filename = None 

        if model_type is CatBoostClassifier:
            filename = "classification_metrics.txt"
        else:
            filename = "regression_metrics.txt"

        with open(filename, "w") as file:
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    file.write(f"{metric}:\n")
                    for key, val in value.items():
                        file.write(f"  {key}: {val}\n")
                else:
                    file.write(f"{metric}: {value}\n")

        print(f"\n--- Метрики модели ({'Классификация' if model_type is CatBoostClassifier else 'Регрессия'}) ---")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}:")
                for key, val in value.items():
                    print(f"  {key}: {val}")
            else:
                print(f"{metric}: {value}")
        print("--- Конец метрик ---\n")

        return metrics
