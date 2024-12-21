from catboost import CatBoostClassifier, CatBoostRegressor

from ml.service.prediction import BrokePredictor

base_classification_model = CatBoostClassifier(
    loss_function="MultiClass",
    silent=False,
)

base_regression_model = CatBoostRegressor(loss_function="RMSE", silent=False)

CLASSIFICATION = BrokePredictor(None, None, model=base_classification_model)
REGRESSION = BrokePredictor(None, None, model=base_regression_model)
