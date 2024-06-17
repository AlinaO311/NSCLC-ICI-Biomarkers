process infer_from_data {
    publishDir "${params.output_dir}",
    mode:'copy',
    saveAs: { fn ->
        fn.endsWith('.csv') ? "Modelling/data/predicted/${fn}" :
        fn.endsWith('.yml') ? "configs/analysis/${fn}" :
        fn
    }
    cpus 8
    memory '32 GB'

    output:
    path "*.yml" , emit: config_analysis
    path "*.csv" , emit: infer_csv

    input:
    val config_file
    val experiment_name
    val data_path
    val infer_outfile 
    val datetime_string
    val output_dir

    script:
    if (experiment_name == "")
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src infer.py --experiment_folder ${config_file} --data_path ${data_path} --output_file $infer_outfile
      PYTHONPATH=$baseDir/bin/src config_script.py --config_path $config_file --output_file $infer_outfile --dir ${params.output_dir}
      """

    else
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src infer.py --experiment_folder ${params.exp_name} --data_path ${data_path} --output_file $infer_outfile
      PYTHONPATH=$baseDir/bin/src config_script.py --config_path $config_file --output_file $infer_outfile --dir ${params.output_dir}
      """

}
