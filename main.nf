ch_train_model_config#!/usr/bin/env nextflow 

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
        preproc_data    : ${params.preproc_data_folder}
        remove_cols     : ${params.cols_to_remove}
        model_type      : ${params.model_type}

        train           : ${params.train}

        infer_from_data : ${params.predict}
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

        (ch_config, ch_data) = fetch_dataset(params.dataset_names, params.datatype, params.mutations_data, datetime_string)
        print(ch_config)
        ch_data.view()
    }
    // otherwise load input files: need DataPrep/*.tsv & config
    else {
        ch_preproc_config = Channel.fromPath("${params.output_dir}/configs/preprocess/*.yml")
    }

    
    // preprocess data sets
    if ( params.preprocess == true ) {
        // if preprocess of dataset needed for categorical processing, creating train/test sets
        (ch_preproc_config, ch_train_config, ch_train_data, ch_test_data)  = preprocess_datasets(ch_config, params.cols_to_remove, params.model_type, datetime_string) 
        ch_train_config.view()
       // PRINT_PATH()
    } // else load previously generated train, test sets
    else {
        ch_train_config = Channel.fromPath("${params.output_dir}/configs/models/*.yml",  checkIfExists: true )
        ch_train_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data_folder}/data/train_data.csv")
        ch_test_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data_folder}/data/test_data.csv")
        print(ch_train_config)
    }
    

    // print warning if both training and pre-trained model are enabled
    //if ( params.train == true && params.predict_models != null ) {
     //   log.warn 'Training is enabled but pre-trained model(s) are also provided, pre-trained models will be ignored'
    //}

    // perform training if specified
    if ( params.train == true ) {
        (ch_train_model_json, ch_model_to_infer_config) = train_data(ch_train_config, datetime_string)
        ch_model_to_infer_config.view()
        }
    else{
        ch_model_to_infer_config = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/config/*.yml",  checkIfExists: true )
        ch_model_to_infer_config.view()
        }
    
    infer_conf = ch_model_to_infer_config.collect()
    test_data = ch_test_data.collect()
    infer_conf.view()

    // perform prediction and inference if specified
    if ( params.infer_from_data == true && params.exp_name == "") {
        output_name = Channel.of("${params.model_type}_prediction_inference.csv")
        (ch_config_for_analysis, ch_infer_csv) = infer_from_data( ch_model_to_infer_config.view() , params.exp_name , ch_test_data.view() , output_name, datetime_string)
        ch_config_for_analysis.view()
    }
    else{
        ch_config_for_analysis = Channel.fromPath("${params.output_dir}/configs/analysis/xgboost_analysis_config.yml",  checkIfExists: true )
        ch_infer_csv = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/inference/*.csv",  checkIfExists: true )
        ch_config_for_analysis.view()
    }
    
    // perform analysis if specified
    if ( params.analyze == true && params.exp_name == "") {
        ch_analysis_out  = analyze_dataset(ch_config_for_analysis , ch_model_to_infer_config.view() , ch_infer_csv.view(), datetime_string)
        ch_analysis_out.view()
    }
    else if ( params.analyze == true && params.exp_name != "") {
        ch_analysis_out  = analyze_dataset(ch_config_for_analysis , params.exp_name , ch_test_data.view())
	ch_analysis_out.view()
    }
    else{
        "No analysis performed"
    }

}

/* 
 * completion handler
 */
workflow.onComplete {
	log.info ( workflow.success ? '\nDone!' : '\nOops .. something went wrong' )
}

