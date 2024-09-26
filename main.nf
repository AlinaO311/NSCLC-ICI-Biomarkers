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
        mutations_data  : ${params.mutations_data}

        visualize	: ${params.visualize}

        test_set_size   : ${params.test_set_size}
        random_seed     : ${params.random_seed}
        preprocess	: ${params.preprocess}

        preproc_data    : ${params.preproc_data}
        remove_cols     : ${params.cols_to_remove}
        model_type      : ${params.model_type}
        train           : ${params.train}

        infer_from_data : ${params.infer_from_data}
        predict_models  : ${params.predict_models}
        predict_train   : ${params.predict_train}
        predict_test    : ${params.predict_test}

        exp_name : ${params.exp_name}

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
    ls -t ${dirPath} | tail -1
    """
}


params.fetch_dataset = params.fetch_dataset ?: true
params.visualize = params.visualize ?: false
params.preprocess = params.preprocess ?: true
params.train = params.train ?: true
params.infer_from_data = params.infer_from_data ?: true
params.analyze = params.analyze ?: true
params.exp_name = params.exp_name ?: false

workflow {
    def expName = null
    datetime_string = GetDateTimeAsString()
     
    mut_file	   = Channel.fromPath("${params.mutations_data}")
    
    // fetch example datasets if none specified
    if ( params.fetch_dataset == true ) {
        // extract gene names from mutations file
        gene_sets = mut_file.flatMap {
            it.readLines().collect { line -> line.tokenize("\t")[0] }
        }

	    mut_file = mut_file.collect()

        (ch_preproc_config, ch_data) = fetch_dataset(params.dataset_names, 
        params.datatype, 
        params.mutations_data, 
        datetime_string, 
        params.test_set_size, 
        params.random_seed)
        println "Preview of fetch dataset harmonized data output channel."
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
     //   ch_train_config.view()
    } // else load previously generated train, test sets
    else {
         def preprocDir = new File("${params.output_dir}/configs/models/")
         def ymlFiles = preprocDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
            return name.toLowerCase().endsWith(".yml")
            }
         })
         if (ymlFiles == null || ymlFiles.length == 0) {
             // Simply continue if the directory does not exist
             println("No preprocess Directory, continuing...")
         } else {
            ch_train_config = Channel.fromPath("${params.output_dir}/configs/models/*.yml", checkIfExists: true)
            ch_train_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data}/data/train_data.csv")
            ch_test_data = Channel.fromPath("${params.output_dir}/Modelling/data/preprocessed/${params.preproc_data}/data/test_data.csv")
         }
    }

    // mod train data
    if (params.train == true) {
        (ch_train_model_json, ch_model_to_infer_config) = train_data(ch_train_config, datetime_string)
        ch_model_to_infer_config.view()
    } else {
        def trainDir = new File("${params.output_dir}/Modelling/output/models/${params.exp_name}/config/")  
        def ymlFiles = trainDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
            return name.toLowerCase().endsWith(".yml")
            }
        })
        if (ymlFiles == null || ymlFiles.length == 0) {
            // Simply continue if the directory does not exist
            println("No training Directory, continuing...")
        } else {
            if (!params.exp_name) {
            base = "${params.output_dir}/Modelling/output/models"
            expName = findMostRecentFile(base)
            println "Using experiment model when train not run with experiment name: ${expName}"
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
                expName = params.exp_name
                ch_model_to_infer_config = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/config/model_config.yml", checkIfExists: true)
                ch_train_model_json = Channel.fromPath("${params.output_dir}/Modelling/output/models/${params.exp_name}/model/model.json", checkIfExists: true)
                ch_model_to_infer_config.view()
            }
        }
    }


    // predict data 
    // 1 ) do we perform inference, yes = true, no = false
    if (params.infer_from_data == true) {
        // 2) if yes, is experiment name empty or not given
        if (!params.exp_name) {
            params.exp_name = ""; // Set default value if not provided
        }
        output_name = Channel.of("${params.model_type}_model_prediction_inference.csv")
        (ch_config_for_analysis, ch_infer_csv) = infer_from_data(ch_model_to_infer_config.view(),
         params.exp_name, 
         ch_test_data.view(), 
         output_name, datetime_string, 
         params.output_dir)
        ch_config_for_analysis.view()
    } else {
       // print("Inference from data not performed")
        def predictedDir = new File("${params.output_dir}/Modelling/data/predicted")
        def csvFiles = predictedDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
            return name.toLowerCase().endsWith(".csv")
            }
	    })
        if (csvFiles == null || csvFiles.length == 0) {
            // Simply continue if the directory does not exist
            println("Directory ${params.output_dir}/Modelling/data/predicted does not exist, continuing...")
        } else {
            ch_config_for_analysis = Channel.fromPath("${params.output_dir}/configs/analysis/xgboost_analysis_config.yml", checkIfExists: true)
            ch_infer_csv = Channel.fromPath("${params.output_dir}/Modelling/data/predicted/*.csv", checkIfExists: true)
            ch_config_for_analysis.view()
            print("Inference from data not performed")
        }
    }
    
    // analyze data
    if (params.analyze == true) {
        if (params.train == true) {
            // Collect the datetime from the channel
                datetime_string
                    .map { it.trim() }
                    .set { collected_datetime }

                // Define a channel with the desired filename format
                ch_exp_name = collected_datetime.map { datetime ->
                    "${params.model_type}_model_${datetime_string}"
                }
                println "Using experiment model following training : ${ch_exp_name.view()}"               
                // Use the extracted value in the analysis
                analyze_dataset(ch_config_for_analysis, ch_exp_name.view(), ch_infer_csv.view(), datetime_string, params.output_dir)
        } else {
            if (!params.exp_name) {
                println "Using given experiment for analysis: ${expName}"
                analyze_dataset(ch_config_for_analysis, expName , ch_infer_csv.view(), datetime_string, params.output_dir)
            } else{
                println "Using given experiment : ${params.exp_name}"
                analyze_dataset(ch_config_for_analysis, params.exp_name, ch_infer_csv.view(), datetime_string, params.output_dir)
            }
        }   
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


