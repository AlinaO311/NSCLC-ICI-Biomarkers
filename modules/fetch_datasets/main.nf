process fetch_dataset {
    publishDir params.output_dir, 
    mode: 'copy',
    saveAs: { fn ->
        fn.endsWith('.tsv') ? "DataPrep/${fn}" : 
        fn.endsWith('.yml') ? "configs/preprocess/${fn}" :
        fn
    }
    
    output:
    path "preprocess_config.yml" , emit: config_harmonization 
    path "*.tsv" , emit: data_harmonization
    path "meta.json"

    input:
    val dataset_names 
    val datatype
    val mutations_data
    val datetime_string

    script:
    if (params.datatype == "categorical" & params.dataset_names == "") 
      """
      fetch_dataset.py --datatype "categorical" --mutations ${mutations_data} --outdir ${params.output_dir} --datetime ${datetime_string}
      """
    
    else if (params.datatype == "numerical" &  params.dataset_names == "" )
      """
      fetch_dataset.py --datatype "numerical" --mutations ${mutations_data} --outdir ${params.output_dir} --datetime ${datetime_string}
      """
    
    else if (params.datatype == "categorical" & params.dataset_names != "")
      """
      fetch_dataset.py --dataset_names ${params.dataset_names} --datatype "categorical" --mutations ${mutations_data} --outdir ${params.output_dir} --datetime ${datetime_string}
      """
    
    else if (params.datatype == "numerical" & params.dataset_names != "")
      """
      fetch_dataset.py --dataset_names ${params.dataset_names} --datatype "numerical" --mutations ${mutations_data} --outdir ${params.output_dir} --datetime ${datetime_string}
      """
    
    else
      """
      echo Check input for fetch_dataset module 
      """
    
}
