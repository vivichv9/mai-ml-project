from catboost import CatBoostClassifier, CatBoostRegressor

from ml.service.prediction import BrokePredictor

base_classification_model = CatBoostClassifier(
    silent=True
)

base_regression_model = CatBoostRegressor(loss_function="MAE", silent=True)

CLASSIFICATION = BrokePredictor(None, None, model=base_classification_model)
REGRESSION = BrokePredictor(None, None, model=base_regression_model)
