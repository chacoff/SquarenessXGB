from xgb_estimator import SquarenessEstimator
import pandas as pd


def round_to_valid(value):
    valid_values = [-40, -35, -30, -25, -20, -15, -10, 0, 10, 15, 20, 25, 30, 35, 40]
    return min(valid_values, key=lambda x: abs(x - value))


def predict_with_params(model, estimator, nutr_vise=None, dim_prof_mont=None, type_prof_lme=None, **other_params):
    """
    Make predictions by finding the encoded value from training data.
    """

    # Find rows in original data that match TYPE_PROF_LME
    matching_rows = estimator._data[estimator._data['TYPE_PROF_LME'].isin(
        estimator._data[estimator._data.index.isin(estimator.x.index)]['TYPE_PROF_LME'].unique()
    )]
    
    # Get the encoded value for TYPE_PROF_LME
    original_data = pd.read_csv(estimator._location, sep=estimator._separator)
    matching_original = original_data[original_data['TYPE_PROF_LME'] == type_prof_lme]
    
    if len(matching_original) == 0:
        raise ValueError(f"TYPE_PROF_LME value '{type_prof_lme}' not found in training data")
    
    # Get the index and find encoded value
    idx = matching_original.index[0]
    if idx in estimator.x.index:
        encoded_type = estimator.x.loc[idx, 'TYPE_PROF_LME']
    else:
        # Re-encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(original_data['TYPE_PROF_LME'].astype(str))
        encoded_type = le.transform([str(type_prof_lme)])[0]
    
    # Create input row
    input_data = pd.DataFrame([{
        'MODELE_PILOTAGE': estimator.x['MODELE_PILOTAGE'].iloc[0],  # Need to handle this too
        'DIM_PROF_MONT': dim_prof_mont,
        'TYPE_PROF_LME': encoded_type,
        'PCRT_PROF_LME': estimator.x['PCRT_PROF_LME'].iloc[0],
        'NUTR_VISE': nutr_vise,
        'NUTR_REAL': estimator.x['NUTR_REAL'].iloc[0],
        'QST_HISTAR_VISE': estimator.x['QST_HISTAR_VISE'].iloc[0],
        'DBT_AILE_INT_SUP': estimator.x['DBT_AILE_INT_SUP'].iloc[0],
        'DBT_AME_SUP': estimator.x['DBT_AME_SUP'].iloc[0],
        'DBT_AILE_INT_SUP_AVANT_EQ': estimator.x['DBT_AILE_INT_SUP_AVANT_EQ'].iloc[0],
        'DBT_AME_SUP_AVANT_EQ': estimator.x['DBT_AME_SUP_AVANT_EQ'].iloc[0],
        'DBT_AILE_INT_SUP_REAL': estimator.x['DBT_AILE_INT_SUP_REAL'].iloc[0],
        'DBT_AME_SUP_REAL': estimator.x['DBT_AME_SUP_REAL'].iloc[0],
    }])
    
    # Override with any additional params
    for key, value in other_params.items():
        if key in input_data.columns:
            input_data[key] = value
    
    predictions = model.predict(input_data)
    
    print(f'MOD_DBT_AILE_SUP = {round_to_valid(predictions[0][0])}')
    print(f'MOD_DBT_AME_SUP = {round_to_valid(predictions[0][1])}')
    
    return predictions



def main():

    estimator = SquarenessEstimator('data\\Collecte_dev.csv', ',', 'models\\model_SquarenessQST')
    estimator.load_data()

    # print("Available columns in estimator.x:")
    # print(estimator.x.columns.tolist())

    model = estimator.load_estimator_model()

    # Check if there's a raw dataframe before feature extraction
    if hasattr(estimator, 'data'):
        print("Available columns in estimator.data:")
        print(estimator.data.columns.tolist())

    results = predict_with_params(
        model=model,
        estimator=estimator,
        nutr_vise=590.0,
        dim_prof_mont=33153,
        type_prof_lme='WF'
    )


if __name__ == "__main__":
    main()