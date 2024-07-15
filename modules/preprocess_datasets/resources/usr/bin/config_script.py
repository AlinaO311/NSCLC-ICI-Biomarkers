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
        default=Path(os.path.join(cwd, "${params.output_dir}", "configs/preprocess/preprocess_config.yml")),
        help="Path to preprocess config file.",
    )
    parser.add_argument('--model_type', help='ML model, default: xgboost',required=False, default='xgboost')
    parser.add_argument('--outdir', help='name of output directory, default: outdir',required=False, default='outdir')

    args = parser.parse_args()
    config_to_load = read_config(args.config_path)
    output_name = config_to_load['output_name']
    time_val = set_datetime.strip()

    latest_file = os.path.join(cwd, args.outdir,'Modelling','data','preprocessed',output_name+'_'+time_val,'data','train_data.csv')

    yaml = ruamel.yaml.YAML()

    if args.model_type == "xgboost":
        yml_dict = """\

                training_name: xgboost_model
                random_seed:
                model: xgboost
                args: {
                        objective: binary:logistic,
                        max_depth: 6,
                        eta: 0.3,
                        tree_method: hist,
                        enable_categorical: True,
                        subsample: 0.8,  
                        colsample_bytree: 0.6
                        }
                preprocessed_data_path:
                gt_column: "DURABLE_CLINICAL_BENEFIT"
        """
        yaml.preserve_quotes = True
        yaml.explicit_start = True
        yaml_dump = yaml.load(yml_dict)
        yaml_dump['random_seed'] = config_to_load['random_seed']
        yaml_dump['preprocessed_data_path'] = latest_file
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

        with open("xgboost_model_config.yml", 'w') as f:
             yaml.dump(yaml_dump, f)
    else:
        yml_dict = { 'training_name': "keras_feed_forward", # Training info
                'random_seed': config_to_load['random_seed'], # sets seed for model initiation
                'model': "keras_feed_forward" , # Model info
                'args': {
                        'nr_of_epochs': 130,
                        'optimizer': "adam",
                        'loss': "binary_crossentropy", 'metrics': ["accuracy", "mean_squared_error"], # Configure the model architecture.
                        'layers': [ # Only dense layers are supported currently.
                                  {
                                    'type': "dense",
                                    'size': 1,
                                    'activation': "relu"},
                                  {'type': "dense",
                                   'size': 10,
                                   'activation': "relu"},
                                  {'type': "dense" ,
                                   'size': 1,
                                   'activation': "sigmoid"
                                  }
                                  ]
                        }, # For binary classification the final layer must be size 1 with sigmoid activation.
             'preprocessed_data_path': latest_file,
             'gt_column': "DURABLE_CLINICAL_BENEFIT" }
        json_string = json.dumps(yml_dict)
        data = yaml.load(json_string)
        # the following sets flow-style for the root level mapping only
        data.fa.set_block_style()
        with open("keras_model_config.yml", 'w') as f:
            yaml.dump(yml_dict, f)

