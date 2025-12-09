from xgb_estimator import SquarenessEstimator
import pandas as pd


def predict_by_parameters(model, estimator, nutr_vise=None, dim_prof_mont=None, type_prof_lme=None):
    """
    Get predictions for records matching the specified parameters.
    Pass None to skip filtering on that parameter.
    """

    data_source = estimator.data if hasattr(estimator, 'data') else estimator.x

    mask = pd.Series([True] * len(data_source), index=data_source.index)
    
    # Apply filters if parameters are provided
    if nutr_vise is not None:
        mask = mask & (estimator.x['NUTR_VISE'] == nutr_vise)
    if dim_prof_mont is not None:
        mask = mask & (estimator.x['DIM_PROF_MONT'] == dim_prof_mont)
    if type_prof_lme is not None:
        mask = mask & (estimator.x['TYPE_PROF_LME'] == type_prof_lme)
    
    filtered_data = estimator.x[mask]
    
    print(f"Found {len(filtered_data)} matching records")
    
    results = []
    for i in filtered_data.index:
        single_input = estimator.x.iloc[[i]]
        predictions = model.predict(single_input)
        
        result = {
            'index': i,
            'MOD_DBT_AILE_SUP': round(float(predictions[0][0]), 1),
            'MOD_DBT_AME_SUP': round(float(predictions[0][1]), 1)
        }
        results.append(result)
        
        print(f'\nRecord index: {i}')
        print(f'MOD_DBT_AILE_SUP = {result["MOD_DBT_AILE_SUP"]}')
        print(f'MOD_DBT_AME_SUP = {result["MOD_DBT_AME_SUP"]}')
    
    return results


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

    results = predict_by_parameters(
        model=model,
        estimator=estimator
    )


if __name__ == "__main__":
    main()