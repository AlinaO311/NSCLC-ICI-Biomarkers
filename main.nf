#!/usr/bin/env nextflow 

/*
 * Copyright (c) 2023, Clinical Genomics.
 * Distributed under the terms of the MIT License.
 */
import groovy.json.JsonSlurper

if(!params.datatype) {
    error("unknown datatype specified --datatype: categorical or numerical")
}

include { fetch_dataset } from './modules/fetch_dataset'
include { preprocess_datasets } from './modules/preprocess_datasets'
include { train_data} from './modules/train_data'
include { infer_from_data } from './modules/infer_from_data'
include { analyze_dataset } from './modules/analyze_dataset'

log.info """\
        NSCLC-ICI pipeline
        ===============
        fetch_dataset  : ${params.fetch_dataset}
        dataset_names   : ${params.dataset_names}
        datatype        : ${params.datatype}

        visualize	: ${params.visualize}

        mutations_data  : ${params.mutations_data}

        preprocess	: ${params.preprocess}
        preproc_data    : ${params.preproc_data}
        remove_cols     : ${params.cols_to_remove}
        model_type      : ${params.model_type}

        train           : ${params.train}

        infer_from_data : ${params.infer_from_data}
        predict_models  : ${params.predict_models}
        predict_train   : ${params.predict_train}
        predict_test    : ${params.predict_test}

        experiment_name : ${params.exp_name}

        analyze         : ${params.analyze}

        output_dir      : ${params.output_dir}
        """
        .stripIndent()

/* 
 * main script flow
 */

process PRINT_PATH {
  debug true
  output:
    stdout
  script:
  """
  echo $PATH
  """
}

process GetDateTimeAsString {
    output:
    stdout 

    script:
    """
    datetime=\$(date +%y%m%d-%H%M%S)
    echo \$datetime
    """
}

process findMostRecentFile {
    input:
    val dirPath

    output:
    stdout

    script:
    """
    ls -t $PWD/${dirPath} | tail -1
    """
}


params.fetch_dataset = params.fetch_dataset ?: true
params.visualize = params.visualize ?: false
params.preprocess = params.preprocess ?: true
params.train = params.train ?: true
params.infer_from_data = params.infer_from_data ?: true
params.analyze = params.analyze ?: true


workflow {
    datetime_string = GetDateTimeAsString()
     
    mut_file	   = Channel.fromPath("${params.mutations_data}")
    
    // fetch example datasets if none specified
    if ( params.fetch_dataset == true ) {
        // extract gene names from mutations file
        gene_sets = mut_file.flatMap {
            it.readLines().collect { line -> line.tokenize("\t")[0] }
        }

	    mut_file = mut_file.collect()

        (ch_preproc_config, ch_data) = fetch_dataset(params.dataset_names, params.datatype, params.mutations_data, datetime_string)
        print(ch_preproc_config)
        ch_data.view()
    }
    // otherwise load input files: need DataPrep/*.tsv & config
    else {
        ch_preproc_config = Channel.fromPath("${params.output_dir}/configs/preprocess/*.yml")
    }
   
    // visualize missing data 
    if (params.visualize) {
        visualize_dataset(ch_data)
    } else {
        print("Skipping visualize process.")
    }
    
    // preprocess data sets
    if ( params.preprocess == true ) {
        // if preprocess of dataset needed for categorical processing, creating train/test sets
        (ch_preproc_config, ch_train_config, ch_train_data, ch_test_data)  = preprocess_datasets(ch_preproc_config, params.cols_to_remove, params.model_type, datetime_string) 
        ch_train_config.view()
    } // else load previously generated train, test sets
    else {
        ch_train_config = Channel.fromPath("${params.output_dir}/configs/models/*.yml",  checkIfExists: true )
        ch_train_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data}/data/train_data.csv")
        ch_test_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data}/data/test_data.csv")
        print(ch_train_config)
    }
    
    // train data   
    if (params.train == true) {
        (ch_train_model_json, ch_model_to_infer_config) = train_data(ch_train_config, datetime_string)
        ch_model_to_infer_config.view()
    } else {
        if (!params.exp_name) {
            base = "${params.output_dir}/Modelling/output/models"
            expName = findMostRecentFile(base)
            ch_model_to_infer_config = expName.map { mostRecentFile ->
                 def fullPath_yml = "${base}/${mostRecentFile.trim()}/config/*.yml"
                 return fullPath_yml
             }           
            ch_train_model_json =  expName.map { mostRecentFile ->
                 def fullPath_json = "${base}/${mostRecentFile.trim()}/model/*.json"
                 return fullPath_json
             }        
            ch_model_to_infer_config.view() 
        } else {
            ch_model_to_infer_config = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/config/*.yml", checkIfExists: true)
            ch_train_model_json = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/model/*.json", checkIfExists: true)
            ch_model_to_infer_config.view()
        }
    }
    
    //infer from data or predict data 
    // 1 ) do we perform inference, yes = true, no = false
    if (params.infer_from_data == true) {
        // 2) if yes, is experiment name empty or not given
        if (!params.exp_name) {
            params.exp_name = ""; // Set default value if not provided
        }
        output_name = Channel.of("${params.model_type}_prediction_inference.csv")
        (ch_config_for_analysis, ch_infer_csv) = infer_from_data(ch_model_to_infer_config.view(), params.exp_name, ch_test_data.view(), output_name, datetime_string)
        ch_config_for_analysis.view()
    } else {
        ch_config_for_analysis = Channel.fromPath("${params.output_dir}/configs/analysis/xgboost_analysis_config.yml", checkIfExists: true)
        ch_infer_csv = Channel.fromPath("${params.output_dir}/Modelling/data/predicted/*.csv", checkIfExists: true)
        ch_config_for_analysis.view()
        print("Inference from data not performed")
    }

    if (params.analyze == true) {
        if (!params.exp_name) {
            ch_exp_name = expName.map { mostRecentFile ->
                def experiment_name = mostRecentFile.trim()
                return experiment_name
            }
            ch_analysis_out = analyze_dataset(ch_config_for_analysis, ch_exp_name.view(), ch_infer_csv.view(), datetime_string)
            ch_analysis_out.view()
        } else {
            ch_analysis_out = analyze_dataset(ch_config_for_analysis, params.exp_name, ch_infer_csv.view(), datetime_string)
            ch_analysis_out.view() }
    } else {
        print("Analysis not performed")
    }
}

/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}
