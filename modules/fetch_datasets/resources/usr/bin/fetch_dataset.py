#!/usr/bin/env python3

# Import packages:
import argparse
import json
import json
import ruamel.yaml
import os
import re
import pandas as pd
import numpy as np
import glob
import difflib
#from datetime import datetime
from itertools import chain
import chardet


path_list = []
patientSets = {}
sampleSets = {}
mutSets = {}
file_inlist = ['data_clinical_patient.txt','data_clinical_sample.txt','data_mutations.txt']
cwd=os.getcwd().split('work', 1)[0]

def fill_na(data):
    for col in data.columns:
        if data[col].isna().any():
            if data[col].dtype == "O":  # Object type (categorical)
                data[col] = data[col].fillna('unknown')
            else:  # Numeric type
               data[col] = data[col].fillna(-1)
    return data

def normalize_text(s):
    if isinstance(s, str):
        # Convert to lowercase
        s = s.lower()
        # Replace non-alphanumeric characters with spaces
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        # Remove extra spaces
        s = re.sub(r'\s+', ' ', s).strip()
    return s

def group_and_replace_phrases(phrases, threshold=0.2):
    # Get unique phrases
    unique_phrases = list(set(phrases))
    groups = []
    visited = set()
    for idx, phrase in enumerate(unique_phrases):
        if idx not in visited:
            group = [phrase]
            visited.add(idx)
            for jdx, other_phrase in enumerate(unique_phrases):
                if jdx != idx and jdx not in visited:
                    # Check if either phrase is a substring of the other
                    if phrase in other_phrase or other_phrase in phrase:
                        group.append(other_phrase)
                        visited.add(jdx)
            groups.append(group)
    # Further grouping similar phrases based on common words
    final_groups = []
    while groups:
        group = groups.pop(0)
        merged = False
        for final_group in final_groups:
            if any(p in group[0] or group[0] in p for p in final_group):
                final_group.extend(group)
                merged = True
                break
        if not merged:
            final_groups.append(group)
    # Removing duplicates within groups
    final_groups = [list(set(group)) for group in final_groups]
    phrase_replacement = {}
    for group in final_groups:
        representative = min(group, key=len)  # Choose the shortest phrase as the representative
        for phrase in group:
            phrase_replacement[phrase] = representative
    return phrase_replacement

def replace_phrases_in_column(df, column_name, phrase_replacement):
    df[column_name] = df[column_name].apply(lambda x: phrase_replacement.get(normalize_text(x), x))
    return df

def log2_normalization(x):
    # Find the minimum value in the series
    min_value = x.min()
    # If the minimum value is less than or equal to zero, calculate the shift required
    shift = -min_value + 1 if min_value <= 0 else 0
    # Apply the shift and then perform log2 normalization
    return np.log2(x + shift + 1)

def normalize(x):
    return (x - x.mean()) / x.std()

class FetchData(object):
    def __init__(self, dataSets, mutationLists):
        	self.data = dataSets
       		self.mutations = mutationLists
    def _get_data(self):
        print("Data directories included:")
        for study in self.data:
            print(study)
            for entry in os.listdir(os.path.join(cwd ,'Data', study)):
                   if entry in file_inlist:
                       path_list.append(os.path.join(cwd , 'Data', study , entry) )
        for file_path in path_list:
            if 'patient' in file_path and os.path.isfile(file_path):
            # Read data files:
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False)
                patientSets[file_path] = df
                print("Successfully read: " + file_path)
            if 'sample' in file_path and  os.path.isfile(file_path):
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False)
                sampleSets[file_path] = df
                print("Successfully read: " + file_path)
            if 'mutations' in file_path and  os.path.isfile(file_path):
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False)
                mutSets[file_path] = df
                print("Successfully read: " + file_path)
        return patientSets, sampleSets, mutSets 
    def _harmonize(self, patientSets, sampleSets, mutSets):
        # read in feature list
        feature_file = open(os.path.join(cwd , 'Data/features_v1.txt'), "r") ###TODO: mod to allow user input? or allow what type specs
        # reading the file 
        feature_read = feature_file.read() 
        # replacing end splitting the text when newline ('\n') is seen, remove empty str. 
        feat_list = list(filter(None, feature_read.split("\n") ))
        C1,C2 =[],[]
        colsList, ls = [],[]
        #get list of col names from patient and sample data
        for i in patientSets.values():
            colsList+=i.columns.values.tolist()
            for k in sampleSets.values():
                colsList+=k.columns.values.tolist()
                for vals in mutSets.values():
                    colsList+=vals.columns.values.tolist()
        feat_list = [x.upper() for x in feat_list]
        colsList = [x.upper() for x in colsList]
        ##### First harmonize data columns
        for df_name, df in patientSets.items():
            # Create temporary lists to hold matching columns
            df.columns = df.columns.str.upper()
            tmp_list_dict = {g: [] for g in feat_list}
            if any('DURABLE' in col for col in df.columns):
            # Drop columns that contain the string 'RESPON' or 'fs_status'
                df = df.drop(columns=[col for col in df.columns if 'RESPON' in col or 'FS_STATUS' in col])
            else:
                # Check if any column contains the string 'RESPON'
                if any('RESPON' in col for col in df.columns):
                    # Replace column names that contain 'RESPON'
                    df.columns = [col.replace('RESPON', 'DURABLE_CLINICAL_BENEFIT') if 'RESPON' in col else col for col in df.columns]
                    # Drop columns that contain the string 'RESPON' or 'FS_STATUS'
                    df = df.drop(columns=[col for col in df.columns if 'RESPON' in col or 'FS_STATUS' in col])
                else:
                    # Check if any column contains the string 'FS_STATUS'
                    if any('FS_STATUS' in col for col in df.columns):
                        # Replace column names that contain 'FS_STATUS'
                        df.columns = [col.replace('FS_STATUS', 'DURABLE_CLINICAL_BENEFIT') if 'FS_STATUS' in col else col for col in df.columns]
                        # Drop columns that contain the string 'FS_STATUS'
                        df = df.drop(columns=[col for col in df.columns if 'FS_STATUS' in col])
                    else:
                        # Return an error if neither 'RESPON' nor 'FS_STATUS' are found
                        print("Error: Neither 'RESPON' nor 'FS_STATUS' found in columns")
                        patientSets.pop(df_name) 
           # Check each column in the data frame
            for col in df.columns:
                for g in feat_list:
                    if 'STAGE' in g.upper() and 'STAGE' in col:
                        tmp_list_dict[g].append(col)
                    else:
                        if g in col and 'STAGE' not in col.upper():
                            tmp_list_dict[g].append(col)
            # Extract the first item from each tmp_list and add to newList
            for g, tmp_list in tmp_list_dict.items():
                if tmp_list:  # Only add if the list is not empty
                    C1.append(g)
                    C2.append(tmp_list[0])
        emp = pd.DataFrame({'key': C1, 'val': C2})
        #keep the col value that has the max ratio if it occurs more than 1 time
        f = emp.apply(lambda x: x.astype(str).str.upper()).drop_duplicates(subset=['key', 'val'], keep='first')
        # create dict from feature:col value pair to rename data cols
        result = f.groupby('key')['val'].apply(list).to_dict()
        #rename some dict keys
        result['SMOKING_HISTORY'] = result.pop('SMOK')
        # add if statement for TMB and HGVSP
        if 'TMB' in feat_list:
            result['TMB']=['TMB']
            for s in colsList:
                if s.startswith('TMB') or s.endswith('TMB'):
                    result['TMB'].append(s)
        else:
            pass
        keep_cols = [key for key, val in result.items()]
        #rename some dict keys
        if 'HGVSP' in feat_list:
            keep_cols+=['HGVSP']
        else:
            pass
        keep_cols+=['STUDY_NAME']
        def change_names(torename, def_dict):
            #change names of columns within dictionaries to match keys created from features
            col_map = {col:key for key, cols in def_dict.items() for col in cols}
            for i in torename:
                columns = torename[i].columns
                torename[i].columns = [x.upper() for x in columns]
                torename[i] = torename[i].rename(columns=col_map)
        def concatinate_dfs(df_dict, filterCols):
            dfs_to_concat, dfs =[], []
            #add study name col
            for df_name in df_dict:
                df = df_dict[df_name]
                df['STUDY_NAME'] = re.search(r'(?<=Data\/)(.*?)(\/data)', df_name).group(1)
            # need to remove possible duplicate name columns
            for df in df_dict.values():
                dfs.append(df)
                for df in dfs:
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
                    dfs_to_concat.append(df)
            concatinated_df = pd.concat(dfs_to_concat, ignore_index = True)
            concatinated_df = concatinated_df.filter(filterCols)
            return concatinated_df
        change_names(patientSets, result)
        change_names(sampleSets, result)
        change_names(mutSets, result)
        ## create unique list from above cols and the feat_list
        keep_feats = list(set(keep_cols + feat_list))
        keep_feats = [s.strip() for s in keep_feats]
        keep_feats.sort(key=len, reverse=True)  # Sort by length in descending order        
        all_clinical_data = concatinate_dfs(patientSets, keep_feats)
        all_clinical_data.rename(columns=lambda x: x.strip(), inplace=True)
        # Sample data:  
        all_sample_data = concatinate_dfs(sampleSets, keep_feats)
        all_sample_data.rename(columns=lambda x: x.strip(), inplace=True)
        all_mut_data = concatinate_dfs(mutSets, keep_feats)
        all_mut_data.rename(columns=lambda x: x.strip(), inplace=True)
        # load mutation collumns
        mut_file = open(os.path.join(cwd , self.mutations), "r")
        mut_list = mut_file.read().translate({ord(c): None for c in "[]'"}).split(',')
        targets = [word for word in mut_list if '=' in word]
        for t in targets:
            ls.append(t.split('=', 1)[0])
        names = [re.sub('.*[\n]+', '', x) for x in ls]
        # for mut in mut_list split into sep list by newlines then join into separate lists, rm empty lists
        m_lst = [ ele for ele in [l.split(',') for l in ','.join(mut_list).split('\n')] if ele != ['']]
        # replace string before = in each element within each list and strip whitespace
        muts = [[re.sub('\w+[=]+', '', y).strip() for y in x] for x in m_lst]
        # check if HGVSP is in feature list
        if 'HGVSP' in keep_feats:
            # create concat column from HUGO_SYMBOL and HGVSP (protein mod/consequence)
            all_mut_data['MUT_HGVSP'] = all_mut_data['HUGO_SYMBOL']+'_'+all_mut_data['HGVSP']
            # Count the occurrence of each Mut/consequence for each Sample_ID
            mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['MUT_HGVSP'], dropna=False).astype(int)
            all_mut_data.drop(['HUGO_SYMBOL','MUT_HGVSP','HGVSP'], axis=1, inplace=True)
        elif 'CONSEQUENCE'  in keep_feats:
            all_mut_data['MUT_CONSEQUENCE'] = all_mut_data['HUGO_SYMBOL']+'_'+all_mut_data['CONSEQUENCE']
            # Count the occurrence of each Mut/consequence for each Sample_ID
            mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['MUT_CONSEQUENCE'], dropna=False).astype(int)
            all_mut_data.drop(['HUGO_SYMBOL','MUT_CONSEQUENCE','CONSEQUENCE'], axis=1, inplace=True)
        else:
            mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['HUGO_SYMBOL'], dropna=False).astype(int)
            all_mut_data.drop('HUGO_SYMBOL', axis=1, inplace=True)
        # if count is greater than 2 set to 1, else 0
        mutDF.iloc[:,1:] = mutDF.iloc[:,1:].applymap(lambda x: x if x >= 1 else 0)
        mutDF = mutDF.loc[:, pd.notnull(mutDF.columns)]
        length=len(names)
        for name in range(length):
             # sum across rows where col matches any of mutations in list muts
            matching_columns = [col for col in mutDF.columns if any(m in col for m in muts[name])]
            mutDF[names[name]] = mutDF.filter(items=matching_columns).sum(1)
        mutDF['TUMOR_SAMPLE_BARCODE'] = mutDF.index
        mutDF.index.name = None
        mutationMerged_dict = all_sample_data.merge(mutDF.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left')
        unique_values = list(set(chain(*muts)))+names  # Get unique values from flattened list
        getCols = [cols for cols in mutationMerged_dict if any(str(cols).startswith(val) for val in unique_values)]
        #### get columns that are that match the feature list and mutation list
        existing_columns_to_keep = [col for col in getCols+keep_feats if col in mutationMerged_dict]
        ## subset mutation df to existing columns to keep 
        subset_df = mutationMerged_dict[existing_columns_to_keep]
        # drop columns that are not in the feature list in order to fill mut cols with na with 0 
        subset_df = subset_df.drop(columns=getCols)
        mutationMerged =subset_df.join(mutationMerged_dict.loc[:,getCols].fillna(0).astype(int))
        ##### Harmonize df values
        if 'TMB' in keep_feats:
            if (mutationMerged_dict['TMB'].dtype != 'float64' and mutationMerged_dict['TMB'].dtype != 'int64') and mutationMerged_dict['TMB'].astype(str).str.contains(',').any():
                # Ensure all values are strings before using .str accessor
                mutationMerged_dict['TMB'] = mutationMerged_dict['TMB'].astype(str)
                # Replace commas with periods
                mutationMerged_dict['TMB'] = mutationMerged_dict['TMB'].str.replace(',', '.')
            else:
                pass
            mutationMerged_dict['TMB'] = pd.to_numeric(mutationMerged_dict['TMB'], errors='coerce').fillna(0).astype(float)
            mutationMerged_dict['TMB_norm'] = mutationMerged_dict.groupby('STUDY_NAME')['TMB'].transform(normalize)
            mutationMerged_dict['TMB_norm_log2'] = mutationMerged_dict.groupby('STUDY_NAME')['TMB'].transform(log2_normalization)
            mutationMerged_dict = mutationMerged_dict.loc[:,['PATIENT_ID','SAMPLE_ID','TMB']].join(mutationMerged_dict.loc[:,getCols].fillna(0).astype(int))
        else:
            mutationMerged_dict = mutationMerged_dict.loc[:,-getCols].join(mutationMerged_dict.loc[:,getCols].fillna(0).astype(int))
        # Combine all data:
        patient_sample_data = pd.merge(all_clinical_data, mutationMerged.merge(all_mut_data.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left') , on=['PATIENT_ID','STUDY_NAME'], how='left')
        ### remove data that is more than 50% missing
        threshold = len(patient_sample_data.columns) / 2
        patient_sample_data.dropna(thresh=threshold, inplace=True)
        df=fill_na(patient_sample_data) 
        ### fill object cols with Missing add -1 to numeric cols
        ### fix Durable clinical benefit entries
        def map_values(val):
            if val == 0:
                return 0
            if val == 1:
                return 1
            if val.startswith('N'):
                return 0
            else:
                return 1
        df['DURABLE_CLINICAL_BENEFIT'] = df['DURABLE_CLINICAL_BENEFIT'].apply(map_values)
        repl_df = df.copy()
        # Process each string column
        for column in df.select_dtypes(include=['object']).columns:
            if column not in ['PATIENT_ID', 'SAMPLE_ID', 'STUDY_NAME']:
                new_values = []   
                for value in df[column]:
                    print(value)
                    try:
                        # Convert to float first to handle numeric strings and floats
                        num = float(value)
                        # Convert float to int for comparison
                        num = int(num)
                        if num <= 1:
                            new_values.append('negative')
                        elif num < 49:
                            new_values.append('weak')
                        elif num > 50:
                            new_values.append('strong')
                        else:
                            new_values.append(value)
                    except (ValueError, TypeError):
                        new_values.append(value)
                # Assign the new values back to the column
                df[column] = new_values
                print(set(df[column].tolist()))
                # Extract all words from the column
                all_phrases = df[column].tolist()
                normalized_phrases = [normalize_text(s) for s in all_phrases]
                # Group and replace phrases
                phrase_replacement = group_and_replace_phrases(normalized_phrases)
                # Replace phrases in the DataFrame
                replace_phrases_in_column(repl_df, column, phrase_replacement)
        return repl_df

def Harmonize(self, *args):
    print("Getting Datasets...")
    clinical_set, sample_set, mutations_set  = FetchData(self, *args)._get_data()
    print("Harmonizing Data...")
    dataframe = FetchData(self, *args)._harmonize(clinical_set, sample_set, mutations_set)
    return dataframe

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Harmonize Datasets')
    ## which studies to use for analysis
    parser.add_argument('--dataset_names', help='Which datasets to use', required=False, type=lambda t: [s.strip() for s in t.split(',')], default='luad_mskcc_2015,nsclc_mskcc_2015,luad_mskcc_2020,nsclc_mskcc_2018')
    # user provided name for each run 
    parser.add_argument('--datatype', help='numerical or categorical', required=True)
    parser.add_argument('--mutations', help='File of mutations of interest', required=False, default='mutations.txt')
    # user provided test_set_size:
    parser.add_argument('--test_set_size', help='test set size, default: 0.2', required=False, default=0.2)
    # user provided seed
    parser.add_argument('--random_seed', help='random seed, default: 42', required=False, default=42)
    parser.add_argument('--outdir', help='output directory, default: results', required=False)
    parser.add_argument('--datetime', required=True)
    args = parser.parse_args()
    datasets = []

    # get stored initial datetime from nf created file
    time_val =  args.datetime
    mydatetime = time_val.strip()
    for study in args.dataset_names:
        print(study)
        datasets.append(study)
    inputdata = Harmonize(datasets, args.mutations)
    print('Features from Data', inputdata.columns.tolist())

#    mydatetime = datetime.now().strftime("%Y%m%d-%H%M")
    # save data
    inputdata.to_csv('data_'+mydatetime+'.tsv' , sep='\t',  index=False)
    yaml = ruamel.yaml.YAML()

    # save config 
    if args.datatype == "numerical":
        config = {
            'preprocessor_name': "main_preprocessor",
            'test_set_size': args.test_set_size ,# Part of data. 1 is all data.
            'random_seed': args.random_seed ,# sets seed for training/test set splits
            'output_name': "numerical_preprocess",

            'data_path': os.path.join(cwd, args.outdir ,'DataPrep','data_'+mydatetime+'.tsv')
        }
        json_string = json.dumps(config)
        data = yaml.load(json_string)
        data.fa.set_block_style()
        with open("preprocess_config.yml", 'w') as f:
            yaml.dump(config, f)
    else:
        config = {
            'preprocessor_name': "non_null_one_hotted",
            'test_set_size': args.test_set_size,
            'random_seed': args.random_seed,
            'output_name': "categorical_preprocess",

            'data_path': os.path.join(cwd , args.outdir  ,'DataPrep','data_'+mydatetime+'.tsv')
        }
        json_string = json.dumps(config)
        data = yaml.load(json_string)
        data.fa.set_block_style()
        with open("preprocess_config.yml", 'w') as f:
            yaml.dump(config, f)
    # save metadata
    meta = {
        'studies_used': datasets ,
        'filename': 'data_'+mydatetime,
        'feature_names': inputdata.columns.tolist(),
        'number_of_entries': len(inputdata),
        'categorical_features': inputdata.select_dtypes(include='string').columns.to_list()
    }
    with open("meta.json", 'w') as f:
        json.dump(meta, f)


