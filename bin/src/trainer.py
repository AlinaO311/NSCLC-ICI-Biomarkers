#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from dataloader import DataLoader
from models import (KERAS_MODEL_NAME, XGBOOST_MODEL_NAME, BaseModel,
                        KerasFeedForward, XGBoost)
from utils import prepare_save_folder, read_config

MODEL_OUTPUT_FOLDER = Path("output/models")


class Trainer:
    """Handles the interaction with different machine learning models.

    Public methods:
    train -- Train the model using the training data.
    predict -- Run inference on the input data set
            and save the intput data together with the predictions.

    Instance variables:
    config -- The path to the model configuration.
    dataloader -- The configured data loader.
    model_type -- The type of machine learning model to load.
    model -- A instance of the BaseModel machine learning class.
    """

    def __init__(
        self,
        config_path: Path,
        resume_model: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        """Initialize the trainer class.

        Arguments:
            config_path -- The path to the model configuration.

        Keyword Arguments:
            resume_model -- The path to an existing model that should be resumed.
                (default: {None})
            data_path -- The path to the input data set. Note! The path to the
                training data is configured using the config file. (default: {None})
        """
        self.config = read_config(config_path)
        
        if "data_path" in self.config and self.config["data_path"] :
            data_path = Path(self.config["data_path"])
        else:
            data_path = Path(self.config["preprocessed_data_path"])
        self.dataloader = DataLoader(data_path, self.config["gt_column"])
        self.dataloader.load_data()

        self.model_type = self.config["model"]
        self.model = self._init_model(resume_model)

    def _init_model(self, model_path: Optional[Path] = None) -> BaseModel:
        """Initialise the machine learning model.

        Keyword Arguments:
            model_path -- The path for loading an existing model. If no path is given
                a new model will be created according to the configuration file.
                (default: {None})
        """
        print("Initiating the model..")
        if self.model_type == XGBOOST_MODEL_NAME:
            return XGBoost(self.config, model_path)
        elif self.model_type == KERAS_MODEL_NAME:
            # Get number of columns in the input data. It is needed when building the model.
            number_of_columns = len(self.dataloader.get_data().columns)

            return KerasFeedForward(self.config, number_of_columns, model_path)
        else:
            assert (
                False
            ), "ERROR: No matching model type found! Check spelling in the model config file?"

    def _save_trained_model(self) -> None:
        """Save the machine learning model to disk."""
        # Prepare save folder.
        output_dir = prepare_save_folder(
            MODEL_OUTPUT_FOLDER,
            self.config["training_name"],
            ["config", "model"],
            {"model_config": self.config},
        )

        # Save the model in the model directory.
        self.model.save_model(os.path.join(output_dir, "model"))
        print(f"Saved the model artifacts to {output_dir}")

    def _save_prediction(
        self,
        data: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_true: pd.DataFrame,
        output_file: Path,
    ) -> None:
        """Save the predictions of the model (together with the input data set) to disk.

        Arguments:
            data -- The input data set (predictor data set).
            y_pred -- The predicted values.
            y_true -- The ground truth values.
            output_path -- The path where to save the resulting data set.
        """
        print(f"\nSaving predictions to {output_file}")
        output_file.parent.mkdir(exist_ok=True, parents=True)

        # Save predictions to csv file.
        data[self.config["gt_column"]] = y_true
        data["predicted"] = y_pred
        data.to_csv(output_file, index=False)

    def train(self) -> None:
        """Train the model using the training data (specified in the config file)."""
        x_train = self.dataloader.get_data()
        print(x_train.apply(lambda x: x.unique()))
        y_train = self.dataloader.get_ground_truth()
        print(y_train.unique())

        print("\n\nSummarising training data before training:\n")
        x_train.info(memory_usage=False) # Prints info.
        print(
            f"\nPrinting the first five rows of the training data:\n"
            f"{x_train.head(5)}"
        )

        print("\n--------Training model-------")
        self.model.train(x_train, y_train)
        print("\n---------Finished!----------")

        self._save_trained_model()

    def predict(self, output_file: Path) -> None:
        """Run inference on the input data set and save the results.

        Arguments:
            output_file -- The path to where to save the resulting data set.
        """
        x_test = self.dataloader.get_data()
        y_true = self.dataloader.get_ground_truth()

        print("Performing inference...")
        y_pred = pd.DataFrame(self.model.inference(x_test), index=None, columns=None)

        self._save_prediction(
            data=x_test, y_true=y_true, y_pred=y_pred, output_file=output_file
        )

