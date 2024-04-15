#!/usr/bin/env python3


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import os
import eli5
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from tensorflow import keras
from dask.distributed import Client

from dask.dataframe import from_pandas
import dask_ml.model_selection as dms


XGBOOST_MODEL_NAME = "xgboost"
KERAS_MODEL_NAME = "keras_feed_forward"

class BaseModel(ABC):
    """A base class for machine learning models.

    Public methods:
    train -- Trains the model on the training data.
    inference -- Run inference on the predictor data set.
    save_model -- Save the model at the specified location.
    load_model -- Loads an existing model from a directory.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Method for training the model."""

    @abstractmethod
    def inference(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """Method for inference (prediction) on a trained model."""

    @abstractmethod
    def save_model(self, folder: Path) -> None:
        """Method to save the existing model to disk."""

    @abstractmethod
    def _load_model(self, model_path: Path) -> Any:
        """Internal method for loading an previously trained model from disk."""


class XGBoost(BaseModel):
    """An wrapper for the XGBoost model.

    Public methods:
    train -- Trains the model on the training data.
    explain_weights -- Returns an eli5 explanation of the model.
    inference -- Run inference on the predictor data set.
    save_model -- Save the model at the specified location.
    _load_model -- Loads an existing model from a directory.

    Instance variables:
    config -- The configuration used to initialize the model.
    model -- The internal XGBoost model.
    """

    MODEL_FILE_NAME = "model.json"

    def __init__(self, config: dict, model_dir_path: Optional[Path] = None) -> None:
        """Initialize a XGBoost model.

        Arguments:
            config -- Configuration for the model.

        Keyword Arguments:
            model_dir_path -- Path to a stored XGBoost model. If given, the model
                            will be loaded from disk.
        """
        self.config = config

        if model_dir_path:
            print(f"Loading model from: { os.path.join(model_dir_path , self.MODEL_FILE_NAME)}")
            self.model = self._load_model(model_dir_path)
        else:
            print("Creating new model.")
            self.model = xgb.XGBClassifier(
                **self.config["args"],
                random_state=self.config["random_seed"],
            )

        super().__init__()

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the model on the given training data.

        Arguments:
            x_train -- The training data set.
            y_train -- The corresponding ground truth.
        """
        ## create dask Data Matrix with Client 
        client = Client(n_workers=10, threads_per_worker=2, memory_limit='5GB')
        print('starting client ', client)
        total_memory = x_train.memory_usage(index=True).sum()
        print('here is x_train size', total_memory)
        print('starting client ', client)

        # Desired max size per partition (1GB)
        max_partition_size = 1073741824
        parts_split = (int(total_memory) + int(max_partition_size) - 1) // (max_partition_size)  # Ceiling division
        print('number of partitions', parts_split)
        ddf = from_pandas(x_train, npartitions=parts_split)
        print('ddf partitions ', ddf.npartitions)
        y_ddf = from_pandas(y_train, npartitions=parts_split)

        dtrain = xgb.dask.DaskDMatrix(client, ddf, y_ddf)

        # Perform K-fold CV using xgb.dask.cv with custom folds
        kf = dms.model_selection.KFold(n_splits=10, shuffle=True, random_state=self.config["random_seed"])

        # Placeholder for cross-validation scores
        cv_scores = []

        for train_index, test_index in kf.split(ddf):
            # Splitting the data
            X_train, X_test = ddf.iloc[train_index], ddf.iloc[test_index]
            y_train, y_test = y_ddf.iloc[train_index], y_ddf.iloc[test_index]

            # Convert to DaskDMatrix
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
            dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

            # Train model
            d_model = xgb.dask.train(client,
                                   params=self.config["args"],
                                   dtrain=dtrain,
                                   num_boost_round=10,  # Adjust based on CV results
                                   **self.config["args"])
            
            print("K fold model trained: " , d_model)

            # Evaluate model
            predictions = xgb.dask.predict(client, d_model, dtest)
            # Assuming a regression problem, replace with appropriate evaluation function
            score = ((y_test - predictions) ** 2).mean().compute()
            cv_scores.append(score)
        
        # Calculate average score across all folds
        average_score = sum(cv_scores) / len(cv_scores)
        print(f"Average Score: {average_score}") 


    def explain_weights(self) -> str:
        """Returns an eli5 explanation of the model.

        Returns:
            The eli5 explination as a string.
        """
        weights = eli5.format_as_text(eli5.explain_weights(self.model))
        return f"Eli5 XGBoost weights\n {weights}"

    def inference(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """Run inference on the predictor data set.

        Arguments:
            x_test -- The predictor data set.

        Returns:
            A Pandas dataframe with the predicted values.
        """
        ddf = from_pandas(x_test, npartitions=3)
        client = Client()
        dtest = xgb.dask.DaskDMatrix(client, ddf)
        print('Here is model used for prediction: ', self.model)
        predictions_dask_array = xgb.dask.predict(client, self.model, dtest)

        # Compute the Dask Array to get the actual predictions as a NumPy array
        predictions_numpy_array = predictions_dask_array.compute()

        # Convert the NumPy array to a Pandas DataFrame
        predictions_df = pd.DataFrame(predictions_numpy_array, columns=['Predictions'])

        return predictions_df

    def save_model(self, directory: Path) -> None:
        """Save the model at the specified location.

        Arguments:
            folder -- The path to the directory where the model will be saved.
        """
        model_storage_path = os.path.join(directory , self.MODEL_FILE_NAME)
        self.model.save_model(model_storage_path)

    def _load_model(self, model_dir_path: Path) -> Any:
        """Loads an existing model from a directory.

        Arguments:
            model_dir_path -- Path to the folder where the model is stored.

        Returns:
            The loaded XGBoost model.
        """
        model_path = os.path.join(model_dir_path , self.MODEL_FILE_NAME)
        assert os.path.isfile(model_path), f"ERROR: No model found at {model_path}"

        model = xgb.XGBClassifier()
        booster = xgb.Booster()
        booster.load_model(model_path)
        model._Booster = booster
        return model


class KerasFeedForward(BaseModel):
    """A keras feed forward model. Currently only dense (fully connected)
    layers are impelemnted.

    NOTE! This model requires the input data to fulfill:
        - No missing values (NaN).
        - All categorical variables must be turned into dummy (one hot encoded) variables.

    Public methods:
    train -- Trains the model on the training data.
    inference -- Run inference on the predictor data set.
    save_model -- Save the model at the specified location.
    _load_model -- Loads an existing model from a directory.

    Instance variables:
    config -- Configuration used to initialise the model.
    model -- The network model instance.
    """

    def __init__(
        self, config: dict, number_of_columns: int, model_path: Optional[Path] = None
    ) -> None:
        """Initializes a Keras Feed forward model from scratch or by loading from file.

        Arguments:
            config -- A dictionary used to configure the model.
            number_of_columns -- The number of columns in the input data.

        Keyword Arguments:
            model_path -- Path to the direcotry containing the saved model. If given,
                        the mdoel will be loaded from disk.
        """
        self.config = config

        if model_path:
            print(f"Loading model from: {model_path}")
            self.model = self._load_model(model_path)
            print("Summarizing the loaded model")
            self.model.summary()
        else:
            print("Creating new model.")
            self._create_model(number_of_columns)

        super().__init__()

    def _create_model(self, number_of_columns: int) -> None:
        """Initialise a new model using the supplied configuration.

        Arguments:
            number_of_columns -- The number of columns in the input data.
        """
        model_conf = self.config["args"]
        layer_conf = model_conf["layers"]

        # Init the random seed.
        initializer = keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0, seed=self.config["random_seed"]
        )

        # Build the model.
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(number_of_columns,)))
        for layer in layer_conf:
            assert (
                layer["type"] == "dense"
            ), f"Error: Layer {layer['type']} not implemented!"
            self.model.add(
                keras.layers.Dense(
                    layer["size"],
                    activation=layer["activation"],
                    kernel_initializer=initializer,
                )
            )

        # Compile the model.
        self.model.compile(
            optimizer=model_conf["optimizer"],
            loss=model_conf["loss"],
            metrics=model_conf["metrics"],
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the model on the given training data.

        NOTE! The supplied data set must not contain any NaNs and all
            categorical variables transformed into dummy variables.

        Arguments:
            x_train -- The training data set.
            y_train -- The ground truth data.
        """
        self.model.fit(
            x_train.to_numpy(),  # Keras wants numpy arrays.
            y_train.to_numpy(),
            epochs=self.config["args"]["nr_of_epochs"],
        )

    def inference(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """Run inference on the predictor data set.

        NOTE! The supplied data set must not contain any NaNs and all
            categorical variables transformed into dummy variables.

        Arguments:
            x_test -- The predictor data set.

        Returns:
            A Pandas dataframe with the predicted values.
        """
        y_pred = self.model.predict(
            x_test.to_numpy(),  # Keras wants numpy arrays.
            batch_size=None,
            verbose="auto",
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        classes = np.round(y_pred).astype(int)
        return classes

    def save_model(self, directory: Path) -> None:
        """Save the model at the specified location.

        Arguments:
            directory -- The path to the directory where the model will be saved.
        """
        self.model.save(directory)

    def _load_model(self, model_path: Path) -> Any:
        """Loads an existing model from a directory.

        Arguments:
            model_dir_path -- The path to the directory where the model is stored.

        Returns:
            The loaded Keras feed forward model.
        """
        return keras.models.load_model(model_path)

