import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import onnxmltools


class SquarenessEstimator:
    def __init__(self, location: str, separator: str, name: str):
        super().__init__()
        self._data: pd = None
        self._location: str = location
        self._separator: str = separator
        self.features: list[str] = [
            # 'CLE_QST',
            'MODELE_PILOTAGE',
            # 'DTE_CREA_REC',
            # 'CO_ORIG_COUL',
            # 'AA_COUL',
            # 'NUM_COUL',
            # 'NUM_BB',
            # 'TYPE_BB',
            # 'TYPE_PROF_MONT',
            'DIM_PROF_MONT',
            'TYPE_PROF_LME',
            # 'DIM_PROF_LME',
            'PCRT_PROF_LME',
            'NUTR_VISE',
            'NUTR_REAL',
            'QST_HISTAR_VISE',
            # 'CMPGE_LMG',
            # 'MONTAGE',
            # 'NUM_COURANT',
            'DBT_AILE_INT_SUP',
            'DBT_AME_SUP',
            'DBT_AILE_INT_SUP_AVANT_EQ',
            'DBT_AME_SUP_AVANT_EQ',
            'DBT_AILE_INT_SUP_REAL',
            'DBT_AME_SUP_REAL',
            # 'MODE_ENF_DP',
            # 'CORR_EQ_AILE_SUP_REAL',
            # 'CORR_EQ_AME_SUP_REAL',
            # 'CLE_STAT',
            # 'NBRE_DEF16_BM',
            # 'NBRE_BF'
        ]
        self.targets: list[str] = [
            'MOD_DBT_AILE_SUP',
            'MOD_DBT_AME_SUP'
        ]
        self.categorical_cols: list[str] = [
            'MODELE_PILOTAGE',
            # 'DTE_CREA_REC',
            # 'TYPE_BB',
            # 'TYPE_PROF_MONT',
            'TYPE_PROF_LME',
            # 'CMPGE_LMG',
            # 'MONTAGE',
            # 'MODE_ENF_DP'
        ]
        self.na_cols: list[str] = [
            'MOD_DBT_AILE_SUP',
            'MOD_DBT_AME_SUP'
        ]
        self.x: pd = None
        self.y: pd = None
        self._name: str = name

    def load_data(self) -> None:
        """ load data in a panda dataframe and follows to drop NA values and encoder categorical values"""
        self._data = pd.read_csv(self._location, sep=self._separator)
        self._drop_nas()
        self._encode_labels()

        _x = self._data[self.features].copy()
        _y = self._data[self.targets].copy()

        # @TODO: work around to export to onnx. Pay atttention with name = f"{old_name}_{i}"
        new_column_names = {old_name: f"f{i}" for i, old_name in enumerate(_x.columns)}
        _x.rename(columns=new_column_names, inplace=True)

        new_column_names = {old_name: f"f{i}" for i, old_name in enumerate(_y.columns)}
        _y.rename(columns=new_column_names, inplace=True)

        self.x = _x
        self.y = _y

    def _drop_nas(self) -> None:
        """ drop na values"""
        self._data = self._data.dropna(subset=self.na_cols)

    def _encode_labels(self) -> None:
        """ Encode categorical columns """
        label_encoders = {}
        for col in self.categorical_cols:
            label_encoders[col] = LabelEncoder()
            self._data[col] = label_encoders[col].fit_transform(self._data[col].astype(str))

    def split_train_test(self, test_size: float, random_state: int, shuffle: bool) -> tuple:
        """ returns X_train, X_test, Y_train, Y_test """

        _x_train, _x_test, _y_train, _y_test = train_test_split(self.x, self.y,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                shuffle=shuffle)
        _shape: int = _x_train.shape[1]

        return _x_train, _x_test, _y_train, _y_test, _shape

    def create_model_xgb(self,
                         objective: str,
                         lr_rate: float,
                         alpha: int,
                         iterations: int,
                         tree_method: str,
                         eval_metric: str,
                         subsample: float,
                         ear_stop: int,
                         _x_train: pd,
                         _x_test: pd,
                         _y_train: pd,
                         _y_test: pd,
                         debug: bool) -> xgb:
        """ return and save the model trained xgboost """

        _model = xgb.XGBRegressor(
            objective=objective,
            learning_rate=lr_rate,
            alpha=alpha,
            n_estimators=iterations,
            tree_method=tree_method,
            eval_metric=eval_metric,
            subsample=subsample,
            early_stopping_rounds=ear_stop
        )

        # train = xgb.DMatrix(x_train, y_train)
        # test = xgb.DMatrix(x_test, y_test)

        _model.fit(_x_train, _y_train, eval_set=[(_x_train, _y_train), (_x_test, _y_test)], verbose=debug)
        results = _model.evals_result()
        _model.save_model(f'{self._name}.json')

        return _model

    def predict_data(self) -> xgb:
        """ load the stored JSON model to make testing predictions """

        _model = xgb.XGBRegressor()
        _model.load_model(f'{self._name}.json')
        print(f'Best iteration: {_model.best_iteration}')

        return _model


def main() -> None:
    estimator = SquarenessEstimator('data\\Collecte_train.csv', ',', 'model_SquarenessQST')
    estimator.load_data()

    x_train, x_test, y_train, y_test, shape = estimator.split_train_test(test_size=0.25, random_state=42, shuffle=True)

    re_train: bool = False
    if re_train:
        model = estimator.create_model_xgb(
            'reg:squarederror',
            0.5,
            9,
            3000,
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
        model = estimator.predict_data()

    input_test: bool = True
    if input_test:
        # test_single_data = x_test.iloc[[9058]]  # 9058, 2753
        # print(f'INPUT DATA:\n{test_single_data.transpose()}')

        tester = SquarenessEstimator('data\\Collecte_dev.csv', ',', 'Test')
        tester.load_data()
        single_input = tester.x.iloc[[1067]]
        print(f'INPUT DATA:\n{single_input.transpose()}')
        predictions = model.predict(single_input)

        print(f'\nMOD_DBT_AILE_SUP = {round(float(predictions[0][0]), 4)}\n'
              f'MOD_DBT_AME_SUP = {round(float(predictions[0][1]), 4)}')

    onnx: bool = False
    if onnx:
        initial_types = [('float_input', onnxmltools.convert.common.data_types.FloatTensorType([None, shape]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)
        onnxmltools.utils.save_model(onnx_model, './model_SquarenessQST.onnx')

        metadata = {"function": "regression", }
        with open('./metadata.json', mode='w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    main()
