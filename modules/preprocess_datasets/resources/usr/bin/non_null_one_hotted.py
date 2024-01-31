#!/usr/bin/env python3

from typing import Tuple
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
    print(set(df['PFS_STATUS']))

    # Drop nan values.
    #print("\n---Summarising data before removing nulls.---")
    df.info(memory_usage=False) # Prints info.
    #df = df.dropna(how="any", axis=0)
    # creating instance of one-hot-encoder
    # get categorical data
    catCols = [col for col in df.columns if df[col].dtype=="O" and ('SAMPLE_ID' not in col and 'PATIENT_ID' not in col and 'PFS_STATUS' not in col)]

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
