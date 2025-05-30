#define functions to calculate deal score for both lease and sales data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Quantile helper function
def rf_quantile_prediction(rf_model, X_data, quantile=0.5):
    all_tree_preds = [estimator.predict(X_data) for estimator in rf_model.estimators_]
    all_tree_preds = np.array(all_tree_preds)
    return np.percentile(all_tree_preds, q=quantile * 100, axis=0)

#Lease data
def compute_lease_deal_scores(lease_df):
    """
    Trains a RandomForest model to estimate lease rate range (5th to 95th percentile)
    and computes a lease deal score for each property. Adds these values to the original dataset.
    
    Parameters:
        lease_df (pd.DataFrame): Input DataFrame with features and 'rate_per_sf_year'.
        
    Returns:
        pd.DataFrame: Original lease_df with 'PredLow', 'PredHigh', and 'Deal_Score' added where possible.
    """

    # Feature columns
    feature_cols = [
        'BUILDING_SF_Modified_Numeric_Final',
        'has_transit_1mi',
        'has_transit_2mi',
        'has_transit_5mi',
        'Office', 'Retail', 'Industrial', 'Healthcare', 'Other',
        'Land', 'Investment', 'Multifamily', 'Hospitality'
    ]
    
    required_cols = ['ID', 'rate_per_sf_year'] + feature_cols
    df_model = lease_df[required_cols].copy()
    df_model.dropna(subset=required_cols, inplace=True)

    # Split data
    X = df_model[feature_cols].values
    y = df_model['rate_per_sf_year'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Compute percentiles
    pred_low = rf_quantile_prediction(model, X, 0.05)
    pred_high = rf_quantile_prediction(model, X, 0.95)

    # Compute deal scores
    deal_scores = []
    for low_val, high_val, actual_rate in zip(pred_low, pred_high, y):
        if actual_rate < low_val:
            score = 10
        elif actual_rate > high_val:
            score = 5
        else:
            center = (low_val + high_val) / 2.0
            delta = (center - actual_rate) / center
            delta_clamped = max(min(delta, 1), -1)
            raw_score = (delta_clamped + 1) / 2
            score = 5 + 5 * raw_score
        deal_scores.append(score)

    # Create a result DataFrame
    result_df = pd.DataFrame({
        'ID': df_model['ID'].values,
        'PredLow': pred_low,
        'PredHigh': pred_high,
        'Deal_Score': deal_scores
    })

    # Merge back to original lease_df on ID
    final_df = lease_df.merge(result_df, on='ID', how='left')

    return final_df

#Sales data
def compute_sale_deal_scores(sale_df):
    """
    Trains a RandomForest model to estimate sale price range (5th–95th percentile)
    and computes a sale deal score for each property. Adds these values to the original dataset.
    
    Parameters:
        sale_df (pd.DataFrame): Input DataFrame with features and 'PRICE_Modified'.
        
    Returns:
        pd.DataFrame: Original sale_df with 'PredLow', 'PredHigh', and 'Deal_Score' added where possible.
    """

    # Feature columns
    feature_cols = [
        'BUILDING_SF_Modified_Numeric_Final',
        'has_transit_1mi',
        'has_transit_2mi',
        'has_transit_5mi',
        'Office', 'Retail', 'Industrial', 'Healthcare', 'Other',
        'Land', 'Investment', 'Multifamily', 'Hospitality'
    ]
    
    required_cols = ['ID', 'PRICE_Modified'] + feature_cols
    df_model = sale_df[required_cols].copy()
    df_model.dropna(subset=required_cols, inplace=True)

    # Split data
    X = df_model[feature_cols].values
    y = df_model['PRICE_Modified'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Compute percentiles
    pred_low = rf_quantile_prediction(model, X, 0.05)
    pred_high = rf_quantile_prediction(model, X, 0.95)

    # Compute deal scores
    deal_scores = []
    for low_val, high_val, actual_rate in zip(pred_low, pred_high, y):
        if actual_rate < low_val:
            score = 10
        elif actual_rate > high_val:
            score = 5
        else:
            center = (low_val + high_val) / 2.0
            delta = (center - actual_rate) / center
            delta_clamped = max(min(delta, 1), -1)
            raw_score = (delta_clamped + 1) / 2
            score = 5 + 5 * raw_score
        deal_scores.append(score)

    # Create a result DataFrame
    result_df = pd.DataFrame({
        'ID': df_model['ID'].values,
        'PredLow': pred_low,
        'PredHigh': pred_high,
        'Deal_Score': deal_scores
    })

    # Merge back to original lease_df on ID
    final_df = sale_df.merge(result_df, on='ID', how='left')

    return final_df