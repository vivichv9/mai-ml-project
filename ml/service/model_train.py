from prediction import BrokeClassification


def main():
    # TODO add data pulling from database
    model = BrokeClassification(None, None)
    model.param_selection()
    model.train()
    model.save_model()


if __name__ == "__main__":
    main()
