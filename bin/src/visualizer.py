#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from comut import comut, fileparsers
import palettable
from datetime import datetime
from dataloader import DataLoader

cwd=os.getcwd().split('work', 1)[0]

set_datetime = os.getenv('DATE_VALUE')
time_val = set_datetime.strip()

class Visualizer:
    """Handles the interaction with the different metric and plot functionality.

    Public methods:
    visualize -- Do the analysis specified in the config tile and save the results
            in the output directory.

    Instance variables:
    output_path -- Where to store the produced output.
    mutfile -- Genes to visualize.
    data_path -- The data that will be visualized.
    """

    def __init__(
        self,
        output_path: Path,
        mutfile: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        """Initialize the analyzer class.

        Arguments:
            output_path -- The path to where to save the results.

        Keyword Arguments:
            data_path -- The path to the data that should be visualized (default: {None})
        """
        self.output_path = output_path
        # Load data directly using pandas as a tab-separated file
        if data_path is not None:
            self.data = pd.read_csv(data_path, delimiter='\t')
        else:
            self.data = None
            print("No data path provided.")
        # Debug print to check if data is loaded
        self.mutfile = mutfile

    def _prepare_save_folder(self) -> Path:
        """Prepares a folder to store plots."""
        output_folder = os.path.join(self.output_path , "visualization"+"_"+time_val)
        Path(output_folder).mkdir(exist_ok=True, parents=True)
        return output_folder

    def _visualize_missing(self, save_path: Path) -> None:
        """
        Arguments:
            data -- The data to visualize.
        Returns:
            A tuple with the training and test data sets.
        """
        keywords = ['stop','start','variant','inframe']
        # Remove columns where the name contains any string from the list
        columns_to_drop = [col for col in self.data.columns if any(substring in col for substring in keywords)]
        # Drop the selected columns
        df_cleaned = self.data.drop(columns=columns_to_drop)
        # Create a boolean DataFrame where True is null
        all_patient_sample_data_filtered = df_cleaned.replace('', pd.NA)
        null_values = all_patient_sample_data_filtered.isnull()
        # Set up the matplotlib figure
        plt.figure(figsize=(35, 8))
        # Draw a heatmap with the boolean values and no cell labels
        sns.heatmap(null_values, cbar=False, yticklabels=False, cmap='Purples')
        plt.title("Heatmap of Null Values in DataFrame")
        plt.xticks(rotation=45, ha='right')  # Tilt the column labels to 45 degrees
        plt.tight_layout()
        plt.show()
        return plt.savefig( os.path.join(save_path , "missing_data_heatmap.png"))

    def visualize(self) -> None:
        """Perfom the visualization and save the
        results in the specified output location."""
        # Prepare save folder
        visualization_output_dir = self._prepare_save_folder()
        # Plot missing data.
        print("\n-----Plotting missing data heatmap----")
        self._visualize_missing( visualization_output_dir)


        print(f"\nImages saved to {visualization_output_dir}")
