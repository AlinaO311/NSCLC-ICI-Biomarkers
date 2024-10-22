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
from comut import comut
from comut import fileparsers
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
            self.data = pd.read_csv(data_path, sep='\t')
            print("Loaded data:", self.data.head())  # Debug print to check if data is loaded correctly
        else:
            self.data = None
            print("No data path provided.")
        # Debug print to check if data is loaded
        print("Loaded data:", self.data)
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
        df_cleaned = data.drop(columns=columns_to_drop)
        # Create a boolean DataFrame where True is null
        all_patient_sample_data_filtered = self.data.replace('', pd.NA)
        null_values = all_patient_sample_data_filtered.isnull()
        # Set up the matplotlib figure
        plt.figure(figsize=(35, 8))
        # Draw a heatmap with the boolean values and no cell labels
        sns.heatmap(null_values, cbar=False, yticklabels=False)
        plt.title("Heatmap of Null Values in DataFrame")
        plt.show()
        return plt.savefig( os.path.join(save_path , "missing_data_heatmap.png"))

    def _plot_comut_variant(self, save_path: Path) -> None:
        """
        Arguments:
            save_path -- The path to save visualization.
        Returns:
            An image of the comut variant plot.
        """
        # Read feature file
        #feature_file = open(os.path.join(cwd , self.featurefile), "r")
        #feature_read = feature_file.read()
        # Add study name to feature list
        # Split the 'type' column into two parts based on '_'
        with open(os.path.join(cwd , self.mutfile), "r") as mut_file:
	        mut_list = mut_file.read().translate({ord(c): None for c in "[]'"}).split(',')
        	targets = [word for word in mut_list if '=' in word]
        ls = []
        for t in targets:
            ls.append(t.split('=', 1)[0])
        # for mut in mut_list split into sep list by newlines then join into separate lists, rm empty lists
        m_lst = [ ele for ele in [l.split(',') for l in ','.join(mut_list).split('\n')] if ele != ['']]
        # replace string before = in each element within each list and strip whitespace
        muts = [[re.sub('\w+[=]+', '', y).strip() for y in x] for x in m_lst]
        # flatten list of lists
        flatmuts = list(itertools.chain.from_iterable(muts))
        # Drop columns that dont start with our selected mutation strings
        mask = self.data.columns.str.startswith(tuple(flatmuts))
        mdata = self.data.loc[:, mask]
        # add back common column ID
        mdata['PATIENT_ID'] = self.data['PATIENT_ID']
        # Select only the relevant columns
        simplified = mdata[['PATIENT_ID', 'gene', 'mutation', 'value']]
        simplified['sample'] = ['sample' + str(i) for i in range(1, len(simplified) + 1)]
        dfset = simplified[['sample', 'gene', 'mutation', 'value']]
        dfset.columns = ['sample', 'category', 'value', 'counts']
        ########## comut plot
        dpi = 300 # change the output resolution
        # You can provide a list of samples to order your comut (from left to right). If none is provided, it will be calculated from your MAF.
        #samples = dfset['sample'].to_list()
        # mapping of mutation type to color. Only these mutation types are shown. Can be any valid matplotlib color, e.g. 'blue', #ffa500, or (1,1,1).
        vivid_10 = palettable.cartocolors.qualitative.Vivid_10.mpl_colors
        num_genes =len(set(dfset['category'].to_list()))
        mut_mapping = {key: vivid_10[i] for i, key in enumerate(set(dfset['value']).to_list())}
        toy_comut = comut.CoMut()
        toy_comut.add_categorical_data(dfset, name = 'Mutation type', category_order = muts, mapping = mut_mapping, tick_style = 'italic')
        side_mapping = {'Mutated samples': 'darkgrey'}
        toy_comut.add_side_bar_data(dfset, paired_name = 'Mutation type', name = 'Mutated samples', position = 'left', 
                                mapping = side_mapping, xlabel = 'Mutated samples')
        toy_comut.plot_comut(figsize = (10,3))
        toy_comut.add_unified_legend()
        print('here is dfset', dfset)
        return toy_comut.figure.savefig( os.path.join(save_path , "mutation_comut_clinical_bar_side.png"), bbox_inches = 'tight', dpi = dpi)


    def visualize(self) -> None:
        """Perfom the visualization and save the
        results in the specified output location."""
        # Prepare save folder
        visualization_output_dir = self._prepare_save_folder()
        # Plot missing data.
        if "variant" in self.data.columns:
            print("\n-----Plotting COMUT and missing data heatmap----")
            self._plot_comut_variant( visualization_output_dir)
            print("\n-----Plotting missing data heatmap----")
            self._visualize_missing( visualization_output_dir)
        else:
            print("\n-----Plotting missing data heatmap----")
            self._visualize_missing( visualization_output_dir)


        print(f"\nImages saved to {visualization_output_dir}")
