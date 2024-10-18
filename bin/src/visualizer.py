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
from dataloader import DataLoader

cwd=os.getcwd().split('work', 1)[0]

class Visualizer:
    """Handles the interaction with the different metric and plot functionality.

    Public methods:
    visualizee -- Do the analysis specified in the config tile and save the results
            in the output directory.

    Instance variables:
    output_path -- Where to store the produced output.
    dataloader -- The configured data loader.
    """

    def __init__(
        self,
#        config_path: Path,
        output_path: Path,
        data_path: Optional[Path] = None,
    ) -> None:
        """Initialize the analyzer class.

        Arguments:
            output_path -- The path to where to save the results.

        Keyword Arguments:
            data_path -- The path to the data that should be visualized (default: {None})
        """
        self.output_path = output_path
        self.dataloader = DataLoader(data_path, "DURABLE_CLINICAL_BENEFIT")
        self.dataloader.load_data()

    def _visualize_missing(self, data, save_path: Path, output_name: str) -> None:
        """
        Arguments:
            data -- The data to viualize.
        Returns:
            A tuple with the training and test data sets.
        """
        # Create a boolean DataFrame where True is null
        all_patient_sample_data_filtered = data.replace('', pd.NA)
        null_values = all_patient_sample_data_filtered.isnull()
        # Set up the matplotlib figure
        plt.figure(figsize=(35, 8))
        # Draw a heatmap with the boolean values and no cell labels
        sns.heatmap(null_values, cbar=False, yticklabels=False)
        plt.title("Heatmap of Null Values in DataFrame")
        plt.show()
        return plt.savefig( os.path.join(save_path , output_name , ".png"))

    def _plot_comut_variant(self, data, mutfile, save_path: Path) -> None:
        """
        Arguments:
            data -- The data to viualize.
            featurefile -- The features to viualize.
        Returns:
            An image of the comut variant plot.
        """
        # Read feature file
        #feature_file = open(os.path.join(cwd , self.featurefile), "r")
        #feature_read = feature_file.read()
        # Add study name to feature list
       # mut_list = list(filter(None, feature_read.split("\n") )).append("STUDY_NAME")
       # load mutation collumns
        mut_file = open(os.path.join(cwd , mutfile), "r")
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
        # Drop columns that dont start with mutation list strings
        mask = data.columns.str.startswith(tuple(flatmuts))
        mdata = data.loc[:, mask]
        mdata['PATIENT_ID'] = data['PATIENT_ID']
        ############################
        # Melt the DataFrame to reshape it
        melted = pd.melt(mdata, id_vars=['PATIENT_ID'], var_name='category', value_name='value')
        # Split the 'type' column into two parts based on '_'
        melted[['gene', 'mutation']] = melted['category'].str.split('_', n=1, expand=True)
        # Select only the relevant columns
        simplified = melted[['PATIENT_ID', 'gene', 'mutation', 'value']]
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
        return toy_comut.figure.savefig( os.path.join(save_path , "mutation_comut_clinical_bar_side" , ".png"), bbox_inches = 'tight', dpi = dpi)


    def visualize(self) -> None:
        """Perfom the visualization and save the
        results in the specified output location."""
       
        # Prepare save folder
        visualization_output_dir = self.output_path
        data= pd.read_csv(self.dataloader.load_data(), sep="\t", header=0, engine='python')

        # Plot missing data.
        if "variant" in data.columns:
            print("\n-----Plotting COMUT and missing data heatmap----")
            self._plot_comut_variant(visualization_output_dir)
        else:
            print("\n-----Plotting missing data heatmap----")
            self._visualize_missing(visualization_output_dir)


        print(f"\nImages saved to {visualization_output_dir}")
#        return analysis_output_dir
