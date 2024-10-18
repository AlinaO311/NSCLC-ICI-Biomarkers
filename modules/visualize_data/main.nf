process visualize_data {
    publishDir "${params.output_dir}",
    mode:'copy'  

    output:
    stdout

    input:
    val data_path
    val datetime_string
    val output_dir
    
    script:

    if (params.exp_name == "")
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src visualize.py --data_path ${data_path} --dir ${params.output_dir}
      """

    else
      """
      DATE_VALUE='$datetime_string'
      export DATE_VALUE
      PYTHONPATH=$baseDir/bin/src visualize.py  --data_path ${data_path}  --dir ${params.output_dir}
      """

}
