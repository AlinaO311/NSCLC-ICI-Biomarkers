#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads data and returns either predictors, ground truth, or the entire data set.

    Public methods:
    load_data -- Loads the data from the path given at initalization.
    get_data -- Return predictor columns as a Pandas dataframe.
    get_ground_truth -- Return the ground truth column as a Pandas dataframe.
    get_complete_data -- Return the entire data set as a Pandas dataframe.

    Instance variables:
    data_path -- Path to the data on disk.
    gt_column -- Name of the ground truth column.
    data -- The loaded data.
    """

    def __init__(self, data_path: Path, gt_column: str) -> None:
        """Create data loader.

        Keyword arguments:
        data_path -- Path to the data on disk.
        gt_column -- Name of the ground truth column.
        """
        self.data_path = data_path
        self.gt_column = gt_column

    def load_data(self) ->  pd.DataFrame:
        """Load data from the given data path provided at initalization."""
        print("Loading data..")

        # Internal function to convert the column types of categorical data to the
        # pandas type "category". Currently, any columns with the type "object" will
        # be transformed.
        def convert_types(df: pd.DataFrame) -> pd.DataFrame:
            object_columns = df.columns[df.dtypes == "object"].tolist()
            df[object_columns] = df[object_columns].astype("category")
            return df

        self.data = convert_types(pd.read_csv(self.data_path))
        return self.data

    def get_data(self) -> pd.DataFrame:
        """Return predictor columns as a Pandas dataframe."""
        
        return self.data.drop(columns=[self.gt_column])

    def get_ground_truth(self) -> pd.DataFrame:
        """Return the ground truth column as a Pandas dataframe."""
        return self.data[self.gt_column]

    def get_complete_data(self) -> pd.DataFrame:
        """Return the entire data set (both predictor and ground truth)
        as a Pandas dataframe."""
        return self.data

