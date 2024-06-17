#!/usr/bin/env python3

from typing import Tuple

import pandas as pd

from main_preprocess import split_data


def nn_process(config: dict, df: pd.DataFrame, *args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply 'Main preprocess' to the original data and then transforms the data to numerical by:
        - dropping all rows with null (nan) values
        - creating dummy (one hot encoded) variables of any categorical data.

    (This preprocess is based on PREPROCESSING in feature_engineering_preprocessing.R
    in David Lords original repository https://github.com/davidlord/Biomarkers-immuno-lungcancer.)

    Arguments:
        config -- The preprocess configuration,
        df -- The original data.

    Returns:
        A tuple with the training and test data sets as pandas data frames.
    """
    if args:
        print("Selecting columns..")
        df = df.drop(columns=args.remove_cols, axis=1, inplace=False)
    else:
        pass

    df = df[df['PFS_STATUS'].notna()]

    map = {
    '0': 0 ,
    '1': 1}
    for k, v in map.items():
        df.loc[df['PFS_STATUS'].str.startswith(k, na=False), 'PFS_STATUS'] = v

    #Instead of Drop nan values, replace 
    def fill_na(data):
        for col in data.columns:
            if data[col].isna().any():
                if data[col].dtype == "O":  # Object type (categorical)
                    data[col] = data[col].fillna('Missing')
                else:  # Numeric type
                    data[col] = data[col].fillna(-1)
        return data
    
    # Fill NaN values
    df = fill_na(df)
    df = df.drop(columns=['SAMPLE_ID', 'PATIENT_ID','STUDY_NAME'])
    df.info(memory_usage=False) # Prints info.

    catCols = [col for col in df.columns if df[col].dtype=="O" and 'PFS_STATUS' not in col]

    # Create dummy variables (one hot encoding).
    print("\n\n---Creating dummy (one hot encoded) variables.---")
    df = pd.get_dummies(
        df,
        columns=catCols,
        prefix=catCols,
        drop_first=True,
    )

    print("\n---Summarising data after creating dummy variables.---")
    #df.info(memory_usage=False) # Prints info.
    print("\n")

    return split_data(df, config)
