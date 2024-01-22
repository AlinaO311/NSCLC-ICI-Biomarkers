process preprocess_datasets {
    beforeScript 'ln -s $baseDir/bin/src/ .'

    publishDir "${params.output_dir}",
    mode:'copy',
    saveAs: { fn ->
        fn.endsWith('.csv') ? "Modelling/data/preprocessed/*/data/${fn}" :
        fn.endsWith('.yml') ? "configs/models/${fn}" :
        fn
    }

    output:
    path "*.yml" , emit: config_preproc
    path "test_data.csv" , emit: train_data
    path "train_data.csv", emit: test_data

    input:
    val config_file
    val cols_to_remove
    val model_type

    script:
    if (cols_to_remove == "")
      """
      PYTHONPATH=$baseDir/bin/src preprocess.py --config_path $config_file --outdir ${params.output_dir} --model_type ${model_type}
      """

    else if (cols_to_remove != "" )
      """
      PYTHONPATH=$baseDir/bin/src preprocess.py --config_path $config_file --remove_cols ${cols_to_remove} --outdir ${params.output_dir} --model_type ${model_type}
      """


}
