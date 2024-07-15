#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import Optional

from analyzer import Analyzer

cwd=os.getcwd().split('work', 1)[0]

def main(analysis_config: Path, experiment_dir: Path, dir: Path, data_path: Optional[Path] = None) -> None:
    pathlist = os.path.dirname(data_path).split('/')[:-2]
    pathlist = ['/' if x == '' else x for x in pathlist]
    print('analyze path: ', *pathlist[:-1])
    output_path = os.path.join(*pathlist[:-1], dir ,"Modelling/output" ,"analysis" )
    model_path = os.path.join(*pathlist[:-1], dir, "Modelling/output" , "models", experiment_dir, "model/model.json") # json file from exp_name
    model_config_path = os.path.join(*pathlist[:-1] , dir ,"Modelling/output" , "models", experiment_dir, "config/model_config.yml")  # config file from exp_name
    print('model_config_path ', model_config_path)
    print('model_path', model_path)
    analyser = Analyzer(
        analysis_config,
        model_config_path,
        output_path,
        model_path=model_path,
        data_path=data_path,
    )
    analyser.analyse()


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Script to analyze data and/or model. Results are saved "
        "under <experiment_directory>/analysis/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--analysis_config",
        type=Path,
        help="Path to analysis config file."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=False,
        help="Path to the experiment folder of the trained model."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data on which to run the analysis.",
    )
    parser.add_argument("--dir", type=Path, help="Path to the output folder." )

    args = vars(parser.parse_args())

    main(**args)
