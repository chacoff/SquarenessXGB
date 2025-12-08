import json
import onnxmltools

from xgb_estimator import SquarenessEstimator


def main() -> None:
    # Parameters
    re_train: bool = True
    onnx: bool = False

    # training pipeline
    estimator = SquarenessEstimator('data\\Collecte_train.csv', ',', 'models\\model_SquarenessQST')
    estimator.load_data()

    x_train, x_test, y_train, y_test, shape = estimator.split_train_test(test_size=0.20, random_state=42, shuffle=True)

    if re_train:
        model = estimator.create_model_xgb(
            'reg:squarederror',
            0.1,
            9,
            7000,
            'exact',
            'rmse',
            0.9,
            20,
            x_train,
            x_test,
            y_train,
            y_test,
            True
        )
    else:
        model = estimator.load_model()

    if onnx:
        initial_types = [('float_input', onnxmltools.convert.common.data_types.FloatTensorType([None, shape]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)
        onnxmltools.utils.save_model(onnx_model, 'models/model_SquarenessQST.onnx')

        metadata = {"function": "regression", }
        with open('models/metadata.json', mode='w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    main()
