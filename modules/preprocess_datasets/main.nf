process preprocess_datasets {
    cpus 8
    memory '32 GB'

    publishDir "${params.output_dir}",
    mode:'copy',
    saveAs: { fn ->
        fn.endsWith('_model_config.yml') ? "configs/models/${fn}" :
        fn
    }

    output:
    path "Modelling/data/preprocessed/*/config/preprocess_config.yml"
    path "*.yml" , emit: config_preproc
    path "Modelling/data/preprocessed/*/data/test_data.csv" , emit: train_data
    path "Modelling/data/preprocessed/*/data/train_data.csv", emit: test_data

    input:
    val config_file
    val cols_to_remove
    val model_type
    val datetime_string

    script:
    if (cols_to_remove == "")
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src preprocess.py --config_path $config_file --outdir ${params.output_dir} --model_type ${model_type}  
      PYTHONPATH=$baseDir/bin/src config_script.py --config_path $config_file --outdir ${params.output_dir} --model_type ${model_type} 
      """

    else if (cols_to_remove != "" )
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src preprocess.py --config_path $config_file --remove_cols ${cols_to_remove} --outdir ${params.output_dir} --model_type ${model_type}    
      PYTHONPATH=$baseDir/bin/src config_script.py --config_path $config_file --outdir ${params.output_dir} --model_type ${model_type}
      """


}
