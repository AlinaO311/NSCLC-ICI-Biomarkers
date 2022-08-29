import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.dataloader import DataLoader
from src.models import (KERAS_MODEL_NAME, XGBOOST_MODEL_NAME, KerasFeedForward,
                        XGBoost)
from src.plots import (confusion_matrix, histogram, scatter_plot,
                       scatter_tsne_2d)
from src.utils import prepare_save_folder, read_config


class Analyzer:
    """Handles the interaction with the different metric and plot functionality.

    Public methods:
    analyse -- Do the analysis specified in the config tile and save the results
            in the output directory.

    Instance variables:
    output_path -- Where to store the produced output.
    analysis_config -- A configuration specifying the desired metrics and plots.
    model_config -- A model configuration (must fit the provided model)
    dataloader -- The configured data loader.
    model_path -- The path to a stored model.
    model_type -- The type of machine learning model to load.
    """

    def __init__(
        self,
        analysis_config_path: Path,
        model_config_path: Path,
        output_path: Path,
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        """Initialize the analyzer class.

        Arguments:
            analysis_config_path -- The path to the analysis configuration file.
            model_config_path -- The path to the model configuration file.
            output_path -- The path to where to save the results.

        Keyword Arguments:
            model_path -- The path to the directory of an existing model to load
                (default: {None})
            data_path -- The path to the data that should be analysed (default: {None})
        """
        self.output_path = output_path
        self.model_path = model_path
        self.analysis_config = read_config(analysis_config_path)
        self.model_config = read_config(model_config_path)

        self.model_type = self.model_config["model"]

        if not data_path:
            data_path = Path(self.analysis_config["prediction_data_path"])
        self.analysis_config["prediction_data_path"] = str(data_path)

        self.dataloader = DataLoader(data_path, self.model_config["gt_column"])
        self.dataloader.load_data()

    def _init_model(self, model_path: Optional[Path] = None):
        """Initialise the model.

        Keyword Arguments:
            model_path -- The path for loading an existing model. (default: {None})
        """
        print("Initiating the model...")
        if self.model_type == XGBOOST_MODEL_NAME:
            return XGBoost(self.model_config, model_path)
        elif self.model_type == KERAS_MODEL_NAME:
            number_of_columns = len(self.dataloader.get_data().columns)
            return KerasFeedForward(self.model_config, number_of_columns, model_path)
        else:
            assert False, "No model found!"

    def _eli5_model_weights(self, save_path: Path) -> None:
        """Create an eli5 explanation of the model and save the result.

        Arguments:
            save_path -- The path where to save the result.
        """
        model = self._init_model(self.model_path)
        assert model.explain_weights, "Missing model weight explanation function."
        model.explain_weights()
        with open(save_path / "analysis/model_weights_eli5.txt", "w") as f:
            print("Saving eli5 info.")
            f.write(model.explain_weights())

    def _plot_confusion_matrix(
        self,
        save_path: Path,
        output_name: str,
        ground_truth_col: str,
        prediction_col: str,
        plotargs: Dict[str, Any] = {},
    ) -> None:
        """Plot a confusion matrix and save the plot.

        Arguments:
            save_path -- The path where to save the plot.
            output_name -- The desired name of the resulting confusion matrix image.
            ground_truth_col -- The name of the ground truth column.
            prediction_col -- The name of the prediction column,

        Keyword Arguments:
            plotargs -- Any additional arguments for the plot function.
        """
        data = self.dataloader.get_complete_data()

        confusion_matrix(
            ground_truth_labels=data[ground_truth_col],
            predicted_labels=data[prediction_col],
            class_labels=["Non-Responder", "Responder"],
            imshow_kwargs=plotargs,
        )
        print(f"Saving '{output_name}.png' confusion_matrix plot.")
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
        plt.savefig(save_path / f"analysis/{output_name}.png")

    def _plot_scatter_plot(
        self,
        save_path: Path,
        output_name: str,
        x_column: str,
        y_column: str,
        color_column: str,
        plotargs: Dict[str, Any] = {},
    ) -> None:
        """Create a scatter plot of two specified columns and save the result.

        Arguments:
            save_path -- The path where to save the result.
            output_name -- The desired name of the resulting scatter plot image.
            x_column -- The name of the column used for the x-values.
            y_column -- The name of the column used for the y-values.
            color_column -- A color specification.

        Keyword Arguments:
            plotargs -- Any additional arguments for the plot function.
        """
        data = self.dataloader.get_complete_data()
        df = data[[x_column, y_column, color_column]]

        scatter_plot(df.dropna(), x_column, y_column, color_column, plotargs)

        print(f"Saving '{output_name}.png' scatter plot.")
        plt.savefig(save_path / f"analysis/{output_name}.png", bbox_inches="tight")

    def _plot_histogram(
        self,
        save_path: Path,
        output_name: str,
        column: str,
        type: str,
        plotargs: Dict[str, Any] = {},
    ) -> None:
        """Create a histogram plot of a given column and save the result.

        Arguments:
            save_path -- The path where to save the result.
            output_name -- The desired name of the resulting plot image.
            column -- The name of the data column to use for the histogram.
            type -- The type of histogram.

        Keyword Arguments:
            plotargs -- Any additional kwargs to the plot function.
        """
        data = self.dataloader.get_complete_data()

        histogram(data[column].dropna(), type, plotargs)

        print(f"Saving '{output_name}.png' histogram.")
        plt.savefig(save_path / f"analysis/{output_name}.png", bbox_inches="tight")

    def _tsne_2d(
        self,
        save_path: Path,
        output_name: str,
        columns: List[str],
        groupby: str,
        plotargs: Dict[str, Any] = {},
    ) -> None:
        """Create a tsne 2d plot and saves the result.

        Arguments:
            save_path -- The path where to save the result.
            output_name -- The desired name of the resulting plot image.
            columns -- Which columns to use in the TSNE decomposition.
            groupby -- How to group the data (a.k.a. the colors of the data points).

        Keyword Arguments:
            plotargs -- Any additional kwargs to the plot function.
        """
        data = self.dataloader.get_complete_data()

        scatter_tsne_2d(data, columns, groupby, plotargs)

        print(f"Saving '{output_name}' tnse 2d plot...")
        plt.savefig(save_path / f"analysis/{output_name}.png", bbox_inches="tight")

    def _prepare_save_folder(self) -> Path:
        """Prepares a folder to store the metrics and plots in."""
        configs = {
            "analyze_config": self.analysis_config,
        }
        output_dir = prepare_save_folder(
            self.output_path, "analysis", ["config", "analysis"], configs
        )
        return output_dir

    def _save_metrics(self, save_path: Path) -> None:
        """Creates and saves metrics comparing the ground truth and predicted
            values.

        Arguments:
            save_path -- the path to where to save the resulting metrics
        """
        data = self.dataloader.get_complete_data()
        gt = data[self.model_config["gt_column"]]
        pred = data["predicted"]

        # Calculate the metrics
        result = {}
        result["accuracy"] = accuracy_score(gt, pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            gt, pred, pos_label=0, average="binary"
        )
        result["precision"] = precision
        result["recall"] = recall
        result["fscore"] = fscore

        # Save the result.
        with open(save_path / "analysis/metrics.txt", "w") as f:
            print("Saving metrics.")
            f.write(f"Metrics: {json.dumps(result, indent=0)}")

    def analyse(self) -> None:
        """Perfom the analysis according to the config file and save the
        results in the spacified output location."""
        # Prepare save folder
        output_dir = self._prepare_save_folder()
        config_keys = self.analysis_config.keys()

        # Calculate metrics.
        if self.analysis_config.get("metrics", False):
            print("\n-----Calculating metrics.----")
            self._save_metrics(output_dir)

        # Explain model with eli5.
        if self.analysis_config.get("explain_model_weights_eli5", False):
            assert self.model_path, "No model path given."
            print("\n----Performing eli5 analysis.----")
            self._eli5_model_weights(output_dir)

        # Create confusion matrices.
        if "confusion_matrix" in config_keys:
            print("\n-----Plotting confusion matrices.----")
            for matrix_conf in self.analysis_config["confusion_matrix"]:
                self._plot_confusion_matrix(output_dir, **matrix_conf)

        # Create histograms.
        if "histogram" in config_keys:
            print("\n-----Plotting histograms.----")
            for hist in self.analysis_config["histogram"]:
                self._plot_histogram(output_dir, **hist)

        # Create scatterplots.
        if "scatter_plot" in config_keys:
            print("\n-----Plotting scatter plots.----")
            for plot in self.analysis_config["scatter_plot"]:
                self._plot_scatter_plot(output_dir, **plot)

        # Create TNSE 2D plot (only numerical data).
        if "tsne_2d" in config_keys:
            print("\n-----Plotting TSNE_2D plots.----")
            for tsne_conf in self.analysis_config["tsne_2d"]:
                self._tsne_2d(output_dir, **tsne_conf)

        print(f"\nResults saved to {output_dir}")
