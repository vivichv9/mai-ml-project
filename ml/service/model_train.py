from prediction import BrokePredictor

from globals import base_classification_model, base_regression_model


def main():
    # TODO add data pulling from database
    classification_model = BrokePredictor(None, None, base_classification_model)
    classification_model.param_selection()
    classification_model.train()
    classification_model.save_model("classification")

    regression_model = BrokePredictor(None, None, base_regression_model)
    regression_model.param_selection()
    regression_model.train()
    regression_model.save_model("regression")


if __name__ == "__main__":
    main()
