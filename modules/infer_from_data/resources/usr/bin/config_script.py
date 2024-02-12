#!/usr/bin/env python3

import sys
import argparse
import os
import json
import glob
import ruamel.yaml
from ruamel.yaml.scalarstring import SingleQuotedScalarString, DoubleQuotedScalarString
from datetime import datetime
from pathlib import Path
from utils import read_config

cwd=os.getcwd().split('work', 1)[0]


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Script to prepare config for training or predicton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        required=True,
        type=Path,
        default=Path(os.path.join(cwd, "${params.output_dir}", "configs/model/*.yml")),
        help="Path to preprocess config file.",
    )
    parser.add_argument('--output_file', help='name of output file',required=False)

    args = parser.parse_args()
    config_to_load = read_config(args.config_path)
    train_name = config_to_load['training_name']

    yaml = ruamel.yaml.YAML()

    if train_name == "xgboost_model":
        yml_dict = """\
            # List of confusion matrices to plot.
            confusion_matrix: [
                {
                output_name: "confusion_matrix",
                ground_truth_col: "PFS_STATUS",
                prediction_col: "predicted" 
                }      
            ]
            # List of histograms to plot. Any nan values will be dropped.
            histogram: [
                {
                output_name: "age_histogram",
                column: "AGE",
                type: "continuous",
                plotargs: {
                    bins: 30,
                    title: "Diagnosis Age"
                }
            },
            {
            output_name: "smoking_histogram",
            column: "SMOKING_HISTORY",
            type: "categorical",
            plotargs: {
                title: "Smoking history",
                xtick_label_rotation: 45,
            }
            }
            ]
        # List of scatterplots to plot. Any nan values will be dropped.
            scatter_plot: [
                {
                output_name: "age_vs_smoking_history__scatter_plot",
                x_column: "AGE",
                y_column: "SMOKING_HISTORY",
                color_column: "predicted"
                },
            {
            output_name: "MSI_vs_TMB_norm_log2__scatter_plot",
            x_column: "MSI",
            y_column: "TMB_norm_log2",
            color_column: "PFS_STATUS"
            },
            {
            output_name: "Diagnosis_Age_vs_TMB_norm_log2__scatter_plot",
            x_column: "AGE",
            y_column: "TMB_norm_log2",
            color_column: "PFS_STATUS"
            }
        ]
        """
        yaml.preserve_quotes = True
        yaml.explicit_start = True
        yaml_dump = yaml.load(yml_dict)
        with open("xgboost_analysis_config.yml", 'w') as f:
             yaml.dump(yaml_dump, f)
    else:
        yml_dict = """\
            # Analysis configuration.
            # Metrics like Accuracy, Precision, Recall and F1-score.
            metrics: true
            # Eli5 to explain model weights. Must provide a model path.
            # DOES NOT WORK WITH KERAS FEED FORWARD NETWORK.
            explain_model_weights_eli5: false
            # List of confusion matrices to plot.
        confusion_matrix: [
            {
        output_name: "confusion_matrix",
        ground_truth_col: "Treatment_Outcome",
        prediction_col: "predicted" 
        }
    ]
        # List of histograms to plot. Any nan values will be dropped.
        histogram: [
        {
        output_name: "age_histogram",
        column: "Diagnosis_Age",
        type: "continuous",
        plotargs: {
            bins: 30,
            title: "Diagnosis Age"
        }
        }
    ]
    # List of scatterplots to plot. Any nan values will be dropped.
    scatter_plot: [
    {
        output_name: "Diagnosis_Age_vs_TMB_norm_log2__scatter_plot",
        x_column: "Diagnosis_Age",
        y_column: "TMB_norm_log2",
        color_column: "Treatment_Outcome"
    }
    ]
    # List of TSNE reductions to plot. Only works with numerical data.
    tsne_2d: [
    {
    output_name: "tsne_gt",
    groupby: "Treatment_Outcome", # Decides the color separation.
    columns: ["Diagnosis_Age",
              "Pan_2020_compound_muts",
              "DCB_genes",
              "NDB_genes",
              "TMB_norm_log2",
              "Histology_Lung Adenocarcinoma",
              "Histology_Lung Squamous Cell Carcinoma",
              "Histology_Non-Small Cell Lung Cancer",
              "Smoking_Current/Former",
              "Smoking_Former",
                "Smoking_Never",
               "Sex_Male"],
        }
        ]
        """
        yaml.preserve_quotes = True
        yaml.explicit_start = True
        yaml_dump = yaml.load(yml_dict)
        with open("keras_analysis_config.yml", 'w') as f:
            yaml.dump(yml_dict, f)

