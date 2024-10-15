#!/usr/bin/env python3

import sys
import argparse
import os
import json
import glob
import pandas as pd
import ruamel.yaml
from ruamel.yaml.scalarstring import SingleQuotedScalarString, DoubleQuotedScalarString
from datetime import datetime
from pathlib import Path
from utils import read_config

set_datetime = os.getenv('DATE_VALUE')
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
    parser.add_argument("--outdir", type=Path, help="Path to the output folder." )

    args = parser.parse_args()
    config_to_load = read_config(args.config_path)
    train_name = config_to_load['training_name']
    time_val = set_datetime.strip()

    latest_file = os.path.join(os.getcwd(), train_name+'_prediction_inference.csv')
    # Read the tab-separated file
    df = pd.read_csv(latest_file, sep=',')
   
    yaml = ruamel.yaml.YAML() 

    if train_name == "xgboost_model":
        if df.select_dtypes(include=['object', 'category']).empty:
            yml_dict = """\
            # Metrics like Accuracy, Precision, Recall and F1-score.
            metrics: true

            # Eli5 to explain model weights. Must provide a model path.
            explain_model_weights_eli5: true
            # create SHAP plot . Must provide model path.
            shap_plot: [
                {
                output_name: "shap_tree_explainer",
                plotargs: { }
                }
            ]

            # List of confusion matrices to plot.
            confusion_matrix: [
                {
                output_name: "confusion_matrix",
                ground_truth_col: "DURABLE_CLINICAL_BENEFIT",
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
            output_name: "drug_administration_weeks_histogram",
            column: "ADMINISTRATION_OF_DRUG_WEEKS",
            type: "continuous",
            plotargs: {
                title: "Administration of Drug Weeks"
            }
            }
            ]
            # List of scatterplots to plot. Any nan values will be dropped.
            scatter_plot: [
                {
                output_name: "age_vs_smoking_history__scatter_plot",
                x_column: "AGE",
                y_column(s): ,
                color_column: "predicted"
                },
            {
            output_name: "PDL1_EXP_vs_TMB_norm_log2_scatter_plot",
            x_column: "PDL1_EXP", # for now changed to PDL1_EXP from MSI (what is MSI from ?)
            y_column: "TMB_norm_log2",
            color_column: "DURABLE_CLINICAL_BENEFIT"
            },
            {
            output_name: "Diagnosis_Age_vs_TMB_norm_log2_scatter_plot",
            x_column: "AGE",
            y_column: "TMB_norm_log2",
            color_column: "DURABLE_CLINICAL_BENEFIT",
            prediction_data_path: 
            }
            ]
            """
            print("No categorical columns found.")
            smoking_columns = [col for col in df.columns if 'SMOKING' in col]
            print(smoking_columns)
            yaml.preserve_quotes = True
            yaml.explicit_start = True
            yaml_dump = yaml.load(yml_dict)
            yaml_dump['scatter_plot'][-1]['columns'] = smoking_columns
        else:
            # Identify categorical columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Filter columns containing the string 'SMOKING'
            #smoking_columns = [col for col in df.columns if 'SMOKING' in col]
            cat_col = categorical_columns[0]
            print(cat_col)
            yml_dict = """\
            # Metrics like Accuracy, Precision, Recall and F1-score.
            metrics: true

            # Eli5 to explain model weights. Must provide a model path.
            explain_model_weights_eli5: true
            # create SHAP plot . Must provide model path.
            shap_plot: [
                {
                output_name: "shap_tree_explainer",
                plotargs: { }
                }
            ]

            # List of confusion matrices to plot.
            confusion_matrix: [
                {
                output_name: "confusion_matrix",
                ground_truth_col: "DURABLE_CLINICAL_BENEFIT",
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
            }
            ]
            # stacked bar plot for one-hot encoded data
            stacked_bar_plot: [
            {
                output_name: ,
                columns: ,
                title: 
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
            output_name: "PDL1_EXP_vs_TMB_norm_log2_scatter_plot",
            x_column: "PDL1_EXP", # for now changed to PDL1_EXP from MSI (what is MSI from ?)
            y_column: "TMB_norm_log2",
            color_column: "DURABLE_CLINICAL_BENEFIT"
            },
            {
            output_name: "Diagnosis_Age_vs_TMB_norm_log2_scatter_plot",
            x_column: "AGE",
            y_column: "TMB_norm_log2",
            color_column: "DURABLE_CLINICAL_BENEFIT",
            prediction_data_path: 
            }
            ]
            """
            yaml.preserve_quotes = True
            yaml.explicit_start = True
            yaml_dump = yaml.load(yml_dict)
            yaml_dump['stacked_bar_plot'][-1]['output_name'] = cat_col+"stacked_barplot"
            yaml_dump['stacked_bar_plot'][-1]['columns'] = cat_col
            yaml_dump['stacked_bar_plot'][-1]['title'] = cat_col+" stacked bar plot for categorical data"
        yaml.preserve_quotes = True
        yaml.explicit_start = True
        yaml_dump = yaml.load(yml_dict)
        yaml_dump['scatter_plot'][-1]['prediction_data_path'] = latest_file  # Ensure correct placement
        def format_lists_in_block_style_if_colon_found(val):
            """Convert all lists with a ':' in them to block style."""
            if isinstance(val, list):
                for ind, ele in enumerate(val):
                    ele = format_lists_in_block_style_if_colon_found(ele)
                    if isinstance(ele, str) and ':' in ele:
                        val._yaml_format.set_block_style()  # most important
                        # this ScalarString format step is optional if only using ruamel, but mandatory if using pyyaml CLoader.
                        if '"' in ele:  # for readability.
                            ele = ruamel.yaml.scalarstring.SingleQuotedScalarString(ele)
                        else:
                            ele = ruamel.yaml.scalarstring.DoubleQuotedScalarString(ele)
                    val[ind] = ele
            elif isinstance(val, dict):
                for k in val:
                    val[k] = format_lists_in_block_style_if_colon_found(val[k])
            return val
        yaml_dump = format_lists_in_block_style_if_colon_found(yaml_dump)
        with open("xgboost_analysis_config.yml", 'w') as f:
             yaml.dump(yaml_dump, f)
    else:
        yml_dict = """\
            # Analysis configuration.
            # Metrics like Accuracy, Precision, Recall and F1-score.
            metrics: true

            # create SHAP plot . Must provide model path.
            shap_plot: true

            # List of confusion matrices to plot.
        confusion_matrix: [
            {
        output_name: "confusion_matrix",
        ground_truth_col: "DURABLE_CLINICAL_BENEFIT",
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
        }
    ]
    # List of scatterplots to plot. Any nan values will be dropped.
    scatter_plot: [
    {
        output_name: "Diagnosis_Age_vs_TMB_norm_log2__scatter_plot",
        x_column: "AGE",
        y_column: "TMB_norm_log2",
        color_column: "DURABLE_CLINICAL_BENEFIT"
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


