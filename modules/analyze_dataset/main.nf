process analyze_dataset {
    publishDir "${params.output_dir}",
    mode:'copy'

    cpus 8
    memory '32 GB'

    output:
    path "analysis/" , emit: analysis_outputs

    input:
    val config_file
    val experiment_name
    val data_path
    val datetime_string
    
    script:

    if (params.exp_name == "")
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src analyze.py --analysis_config ${config_file} --experiment_dir ${experiment_name} --data_path ${data_path} 
      """

    else
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src analyze.py --analysis_config ${config_file} --experiment_dir ${params.exp_name} --data_path ${data_path} 
      """

}
