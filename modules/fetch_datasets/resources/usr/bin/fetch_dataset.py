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
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
import ssl

nltk.download('wordnet')
nltk.download('all-nltk') #Downloads all packages.  This step is optional


path_list = []
patientSets = {}
sampleSets = {}
mutSets = {}
file_inlist = ['data_clinical_patient.txt','data_clinical_sample.txt','data_mutations.txt']
cwd=os.getcwd().split('work', 1)[0]

def replace_nan(column, placeholder):
    return column.fillna(placeholder)

def fix_text(s):
    """
    - Convert to lowercase
    - Replace non-alphanumeric characters with underscores
    - Replace spaces with underscores
    - Group '5' as 'upstream' and '3' as 'downstream'
    """
    if not isinstance(s, str):
        return s
    # Normalize text
    s = re.sub(r'[^a-z0-9\s]', '_', s.lower())
    s = re.sub(r'\s+', '_', s).strip('_')
    # Group '5' with upstream and '3' with downstream
    if '5' in s:
        return 'upstream'
    elif '3' in s:
        return 'downstream' 
    return s

# alternative replace_with_fixed function
def replace_with_grouped(series, grouped_dict):
    """
    Replace phrases in a series with their longest representative from grouped_dict.
    """
    def replace_value(value):
        for key, values in grouped_dict.items():
            if value in values:
                return key
        return value
    return series.apply(replace_value)

def group_by_seed(words):
    """
    Group phrases based on seed matching after removing spaces/underscores.

    Args:
        words (list): List of phrases to group.

    Returns:
        dict: Dictionary with longest phrase as the key for each group.
    """
    # Preprocess the words: remove spaces and underscores
    normalized = {word: word.replace(" ", "_") for word in words}
    # Initialize the grouping dictionary
    grouped = defaultdict(list)
    # Group words where one is a seed of another
    for word, norm_word in normalized.items():
        matched = False
        for key in grouped:
            if norm_word in normalized[key] or normalized[key] in norm_word:
                grouped[key].append(word)
                matched = True
                break
        if not matched:
            grouped[word].append(word)
    # Replace keys with the longest word in each group
    final_grouped = {}
    for key, group in grouped.items():
        longest_word = max(group, key=len)
        final_grouped[longest_word] = group
    return final_grouped


def group_replace(words):
    # Normalize words for consistent comparisons
    normalized = {word: word.replace(" ", "_").lower() for word in words}
    negation_prefixes = ['non_', 'not_']
    grouped = defaultdict(list)
    # Group words based on substring matches
    for word, norm_word in normalized.items():
        matched = False
        for key, key_norm in [(k, normalized[k]) for k in grouped]:
            # Exclude groups with negation prefixes
            if any(key_norm.startswith(prefix) for prefix in negation_prefixes) or any(norm_word.startswith(prefix) for prefix in negation_prefixes):
                continue
            # Group if one word is a substring of the other
            if norm_word in key_norm or key_norm in norm_word:
                grouped[key].append(word)
                matched = True
                break
        # Add as a new group if no match was found
        if not matched:
            grouped[word].append(word)
    
    # Use the longest word as the representative
    final_grouped = {}
    for key, group in grouped.items():
        longest_word = max(group, key=len)
        final_grouped[longest_word] = group 
    return final_grouped


def log2_normalization(x):
    """Perform log2 normalization on the input series."""
    # Find the minimum value in the series
    min_value = x.min()
    # If the minimum value is less than or equal to zero, calculate the shift required
    shift = -min_value + 1 if min_value <= 0 else 0
    # Apply the shift and then perform log2 normalization
    return np.log2(x + shift + 1)

def normalize(x):
    """Normalize the input series."""
    return (x - x.mean()) / x.std()

class FetchData(object):
    def __init__(self, dataSets, mutationLists, featureFile):
        self.data = dataSets
        self.mutations = mutationLists
        self.features = featureFile
    def _get_data(self):
        for study in self.data:
            for entry in os.listdir(os.path.join(cwd ,'Data', study)):
                   if entry in file_inlist:
                       path_list.append(os.path.join(cwd , 'Data', study , entry) )
        for file_path in path_list:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            if 'patient' in file_path and os.path.isfile(file_path):
            # Read data files:
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False, encoding=encoding)
                patientSets[file_path] = df
                print("Successfully read: " + file_path)
            if 'sample' in file_path and  os.path.isfile(file_path):
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False, encoding=encoding)
                sampleSets[file_path] = df
                print("Successfully read: " + file_path)
            if 'mutations' in file_path and  os.path.isfile(file_path):
                df = pd.read_csv(file_path, comment='#',header=0, delimiter='\t', low_memory=False, encoding=encoding)
                mutSets[file_path] = df
                print("Successfully read: " + file_path)
        return patientSets, sampleSets, mutSets 
    def _harmonize(self, patientSets, sampleSets, mutSets):
        # read in feature list
        feature_file = open(os.path.join(cwd , self.features), "r")
        # reading the file 
        feature_read = feature_file.read() 
        # replacing end splitting the text when newline ('\n') is seen, remove empty str. 
        feat_list = list(filter(None, feature_read.split("\n") ))
        C1,C2,colsList, ls = [],[], [], []
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
            # Change response columns to match DURABLE CLINICAL BENEFIT
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
           # Check each column in the data frame for matches with the feature list
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
        #add cols of interest to list of cols to keep
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
        # check if HGVSP is ,,in feature list
        if 'HGVSP' in keep_feats:
            # Replace NaN in HGVSP with a placeholder and construct MUT_HGVSP
            all_mut_data['HGVSP'] = replace_nan(all_mut_data['HGVSP'], 'unknown')
            # group and replace values in HGVSP
            # create concat column from HUGO_SYMBOL and HGVSP (protein mod/consequence)
            all_mut_data['MUT_HGVSP'] = all_mut_data['HUGO_SYMBOL']+'_'+all_mut_data['HGVSP']
            # Count the occurrence of each Mut/consequence for each Sample_ID
            # Crosstab: Preserve NaN values in counts
            mutDF = pd.crosstab(all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['MUT_HGVSP']).reindex(all_mut_data['TUMOR_SAMPLE_BARCODE'], fill_value=np.nan)
            #mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['MUT_HGVSP'], dropna=False).astype(int)
            all_mut_data.drop(['HUGO_SYMBOL','MUT_HGVSP','HGVSP'], axis=1, inplace=True)
            all_mut_data_cp = all_mut_data.drop_duplicates(inplace=False)
        elif 'CONSEQUENCE'  in keep_feats:
            # Replace NaN in CONSEQUENCE with a placeholder and construct MUT_CONSEQUENCE
         #   all_mut_data['CONSEQUENCE'] = replace_nan(all_mut_data['CONSEQUENCE'], 'unknown')
            # group and replace values in CONSEQUENCE
          #  all_mut_data = all_mut_data.replace(r'lost', 'loss', regex=True)
           # all_mut_data['CONSEQUENCE'] = all_mut_data['CONSEQUENCE'].str.replace(r'\s+', '_', regex=True)
           # all_mut_data_cp = all_mut_data.copy()
            #Step 1: Extract all words from the column
          #  all_phrases = all_mut_data_cp['CONSEQUENCE'].tolist()
            # Step 2: Ensure each sublist is a list of strings
          #  checked_phrases = [sublist if isinstance(sublist, list) else [sublist] for sublist in all_phrases]
           # repl_mult = [['multiple' if isinstance(item, str) and ',' in item else item for item in sublist] for sublist in checked_phrases]
            # Step 3: Flatten the values - skip NaN values
            # If the sublist is iterable (e.g., list), process its items - Join lists or tuples into a string, skip float/NaN   
          #  flattened_phrases = [item if isinstance(item, str) else ' '.join(map(str, item)) for sublist in repl_mult for item in sublist if not pd.isna(item)]
            # Step 4: Group similar values
          #  replacements = group_by_seed(flattened_phrases)
            ###################
            ########### TRY
            def clean_and_standardize(s):

                """
                Clean and standardize text:
                - Convert to lowercase.
                - Replace non-alphanumeric characters with underscores.
                - Remove multiple spaces/underscores.
                - Replace ',' with 'multiple'.
                """
                if not isinstance(s, str):
                    return s
                if ',' in s:
                    return 'multiple'
                s = re.sub(r'[^a-z0-9\s]', '_', s.lower())
                # replace spaces with underscores
                s = re.sub(r'\s+', '_', s).strip('_')
                # Group '5' with upstream and '3' with downstream
                if '5' in s:
                    return 'upstream'
                elif '3' in s:
                    return 'downstream' 
                elif 'stop_retain' in s or 'stop_gain' in s:
                    return 'stop_gain'
                return s
            all_mut_data_cp = all_mut_data.copy()
            # Step 1: Replace NaN with 'unknown'
            all_mut_data_cp['CONSEQUENCE'] = all_mut_data_cp['CONSEQUENCE'].fillna('unknown')
            # Step 2: Clean and standardize the column
            all_mut_data_cp['CONSEQUENCE'] = all_mut_data_cp['CONSEQUENCE'].apply(clean_and_standardize)
            # Step 3: Replace 'lost' with 'loss'
            all_mut_data_cp['CONSEQUENCE'] = all_mut_data_cp['CONSEQUENCE'].replace(r'lost', 'loss', regex=True)
            # Step 4: Group similar phrases
            unique_phrases = all_mut_data_cp['CONSEQUENCE'].dropna().unique().tolist()
            grouped_dict = group_by_seed(unique_phrases)
            # Step 5: Replace phrases with grouped values
            all_mut_data_cp['CONSEQUENCE'] = replace_with_grouped(all_mut_data_cp['CONSEQUENCE'], grouped_dict)
            #############
            all_mut_data_cp['MUT_CONSEQUENCE'] = all_mut_data_cp['HUGO_SYMBOL']+'_'+all_mut_data_cp['CONSEQUENCE']
            # Count the occurrence of each Mut/consequence for each Sample_ID
            mutDF = pd.crosstab( all_mut_data_cp['TUMOR_SAMPLE_BARCODE'], all_mut_data_cp['MUT_CONSEQUENCE']).reindex(all_mut_data_cp['TUMOR_SAMPLE_BARCODE'], fill_value=np.nan)
            all_mut_data_cp.drop(['HUGO_SYMBOL','MUT_CONSEQUENCE','CONSEQUENCE'], axis=1, inplace=True)
            all_mut_data_cp.drop_duplicates(inplace=True)
        else:
            mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['HUGO_SYMBOL']).reindex(all_mut_data['TUMOR_SAMPLE_BARCODE'], fill_value=np.nan)
            all_mut_data.drop('HUGO_SYMBOL', axis=1, inplace=True)
            all_mut_data_cp = all_mut_data.drop_duplicates(inplace=False)
        # if count is greater than 2 set to 1, else 0
        #mutDF.iloc[:,1:] = mutDF.iloc[:,1:].applymap(lambda x: x if x >= 1 else 0)
        # Try dropping columns with NaN values in the column names first
        try:
            mutDF = mutDF.drop(columns=[np.nan])
        except KeyError:
        # If no such columns exist or the operation fails, fall back to the original method
            if mutDF.columns.isnull().any():
                mutDF = mutDF.loc[:, pd.notnull(mutDF.columns)]
        #mutDF = mutDF.drop(columns=[np.nan])
        length=len(names)
        for name in range(length):
             # sum across rows where col matches any of mutations in list muts - with strict matching
            matching_columns =  [col for col in mutDF.columns for m in muts[name] if col.startswith(f"{m}_")]
            mutDF[names[name]] = mutDF.filter(items=matching_columns).sum(1)
        mutDF['TUMOR_SAMPLE_BARCODE'] = mutDF.index
        mutDF.index.name = None
        mutationMerged_dict = all_sample_data.merge(mutDF.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left')
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
        # Combine all data:
        all_clinical_data.drop_duplicates(inplace=True)
        mutationMerged_dict.drop_duplicates(inplace=True)
        # Replace spaces with underscores in all values
        all_clinical_data = all_clinical_data.replace(' ', '_', regex=True)
        # Replace spaces with underscores in all values
        mutationMerged_dict = mutationMerged_dict.replace(' ', '_', regex=True)
        # Replace spaces with underscores in all values
        all_mut_data_cp = all_mut_data_cp.replace(' ', '_', regex=True)
        patient_sample_data = pd.merge(all_clinical_data, mutationMerged_dict.merge(all_mut_data_cp.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left') , on=['PATIENT_ID','STUDY_NAME'], how='left')
        ### fix Durable clinical benefit entries
        def map_values(val):
            # Return NaN if the value is NaN
            if isinstance(val, float) and math.isnan(val):
                return float('nan')
            # Return 0 if the value is 0, 0.0, or starts with 'N'
            if val == 0 or val == 0.0 or (isinstance(val, str) and val.startswith('N')):
                return 0
            # Return 1 for all other cases
            return 1
        df_no_duplicates = patient_sample_data.drop_duplicates()
        df_no_duplicates['DURABLE_CLINICAL_BENEFIT'] = df_no_duplicates['DURABLE_CLINICAL_BENEFIT'].apply(map_values)
        # Replace spaces with underscores in all values
        df_no_duplicates = df_no_duplicates.replace(' ', '_', regex=True)      
        harmonized_df = df_no_duplicates.copy()
        # Process each string column
        for column in df_no_duplicates.select_dtypes(include=['object']).columns:
            if column not in ['PATIENT_ID', 'SAMPLE_ID', 'STUDY_NAME']:
                new_values = []
                # Group and replace phrases for PDL1 column
                if column.startswith('PDL'):
                    for value in df_no_duplicates[column]:
                        try:
                            num = float(value)
                            # Convert to float first to handle numeric strings and floats
                            if num <= 1:
                                new_values.append('negative')
                            elif num <= 49:
                                new_values.append('weak')
                            elif num >= 50:
                                new_values.append('strong')
                            else:
                                new_values.append(value)
                        except (ValueError, TypeError):
                            if '1' in value:
                                new_values.append('negative')
                            elif '49' in value:
                                new_values.append('weak')
                            elif '50' in value:
                                new_values.append('strong')
                            else:
                                new_values.append(value)
                    # Assign the new values back to the column
                    df_no_duplicates[column] = new_values
                #Step 1: Extract all words from the column
                all_phrases = df_no_duplicates[column].tolist()
                # Step 2: Ensure each sublist is a list of strings
                checked_phrases = [sublist if isinstance(sublist, list) else [sublist] for sublist in all_phrases]
                # Step 3: Flatten the values - skip NaN values
                # If the sublist is iterable (e.g., list), process its items - Join lists or tuples into a string, skip float/NaN   
                flattened_phrases = [item if isinstance(item, str) else ' '.join(map(str, item)) for sublist in checked_phrases for item in sublist if not pd.isna(item)]
              #  normalized_phrases = [fix_text(s) for s in flattened_phrases]
                # Step 4: Group similar values
                replacements = group_replace(flattened_phrases)
                harmonized_df[column] = replace_with_grouped(harmonized_df[column], replacements)
        return harmonized_df

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
    parser.add_argument('--features', help='File of features of interest', required=False, default='features.txt')
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
        datasets.append(study)
    inputdata = Harmonize(datasets, args.mutations, args.features)
    print('Features from Data', inputdata.columns.tolist())

    # save data
    inputdata.to_csv('data_'+mydatetime+'.tsv' , sep='\t',  index=False)
    yaml = ruamel.yaml.YAML()

    # save config 
    if args.datatype == "numerical":
        config = {
            'preprocessor_name': "main_preprocessor",
            'test_set_size': float(args.test_set_size) ,# Part of data. 1 is all data.
            'random_seed': int(args.random_seed) ,# sets seed for training/test set splits
            'output_name': "numerical_preprocess",

            'data_path': os.path.join(cwd, args.outdir ,'DataPrep','data_'+mydatetime+'.tsv')
        }
     #   json_string = json.dumps(config)
      #  data = yaml.load(json_string)
      #  data.fa.set_block_style()
        with open("preprocess_config.yml", 'w') as f:
            yaml.dump(config, f)
    else:
        config = {
            'preprocessor_name': "non_null_one_hotted",
            'test_set_size': float(args.test_set_size),
            'random_seed': int(args.random_seed),
            'output_name': "categorical_preprocess",

            'data_path': os.path.join(cwd , args.outdir  ,'DataPrep','data_'+mydatetime+'.tsv')
        }
      #  json_string = json.dumps(config)
      #  data = yaml.load(json_string)
      #  data.fa.set_block_style()
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


