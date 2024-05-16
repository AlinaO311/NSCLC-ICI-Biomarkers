process train_data {
    publishDir "${params.output_dir}",
    mode:'copy',
    saveAs: { fn ->
        fn ? "Modelling/${fn}" :
        fn
    }
    cpus 8
    memory '32 GB'
   
    output:
    path "output/models/*/model/model.json", emit: model_json
    path "output/models/*/config/model_config.yml", emit: config_copy

    input:
    val config
    val datetime_string

    script:
    """
    DATE_VALUE='$datetime_string'
    export DATE_VALUE
    PYTHONPATH=$baseDir/bin/src train.py --config_path  $config 
    """
}
