#!/usr/bin/env python3

# Import packages:
import argparse
import json
import json
import ruamel.yaml
import os
import re
import pandas as pd
import glob
import difflib
#from datetime import datetime
from itertools import chain

path_list = []
patientSets = {}
sampleSets = {}
mutSets = {}
file_inlist = ['data_clinical_patient.txt','data_clinical_sample.txt','data_mutations.txt']
cwd=os.getcwd().split('work', 1)[0]

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
        C1,C2,C3 =[],[],[]
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
        for g in feat_list:
            # for feature in list of desired features
            for col in set(colsList):
                # if the difference in ratios between matching blocks greater than .25
                if difflib.SequenceMatcher(None, g, col).ratio() > 0.25:
                    # check that either word in col starts with feature word OR difference ratio greater than 0.7
                    # append feature, column value, and ratio to lists to create DF emp 
                    if col.startswith(g) or difflib.SequenceMatcher(None, g, col).ratio() > 0.8:
                        C1.append(g)
                        C2.append(col)
                        C3.append(difflib.SequenceMatcher(None, g, col).ratio())
        emp = pd.DataFrame({'key': C1, 'val': C2, 'ratio': C3})
        #keep the col value that has the max ratio if it occurs more than 1 time
        f = emp.groupby('val', group_keys=False).apply(lambda x: x.loc[x.ratio.idxmax()])
        # create dict from feature:col value pair to rename data cols
        result = f.groupby('key')['val'].apply(list).to_dict()
        for v in result.values():
            for feat in v:
                if 'NONSYNONYMOUS' in feat:
                    result['TMB'].append(feat)
                    break
        #rename some dict keys
        result['SMOKING_HISTORY'] = result.pop('SMOK')
        keep_cols = [key for key, val in result.items() if 'NONSYNONYMOUS' not in key]
        keep_cols+=['STUDY_NAME']
        keep_cols+=['HGVSP']
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
        all_clinical_data = concatinate_dfs(patientSets, keep_cols)
        all_clinical_data.rename(columns=lambda x: x.strip(), inplace=True)
        # Sample data:  
        all_sample_data = concatinate_dfs(sampleSets, keep_cols)
        all_sample_data.rename(columns=lambda x: x.strip(), inplace=True)
        all_mut_data = concatinate_dfs(mutSets, keep_cols)
        all_mut_data.rename(columns=lambda x: x.strip(), inplace=True)
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
        # create concat column from HUGO_SYMBOL and HGVSP (protein mod/consequence)
        all_mut_data['MUT_HGVSP'] = all_mut_data['HUGO_SYMBOL']+'_'+all_mut_data['HGVSP']
        # Count the occurrence of each Mut/consequence for each Sample_ID
        mutDF = pd.crosstab( all_mut_data['TUMOR_SAMPLE_BARCODE'], all_mut_data['MUT_HGVSP'], dropna=False).astype(int)
        # if count is greater than 2 set to 1, else 0
        mutDF.iloc[:,1:] = mutDF.iloc[:,1:].applymap(lambda x: x if x >= 1 else 0)
        mutDF = mutDF.loc[:, pd.notnull(mutDF.columns)]
        length=len(names)
        for name in range(length):
            # sum across rows where col matches any of mutations in list muts
            mutDF[names[name]] = mutDF.loc[:, mutDF.columns.str.startswith(tuple(muts[name]))].sum(1)
        mutDF['TUMOR_SAMPLE_BARCODE'] = mutDF.index
        mutDF.index.name = None
        mutationMerged_dict = all_sample_data.merge(mutDF.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left')
        unique_values = list(set(chain(*muts)))+names  # Get unique values from flattened list
        getCols = [cols for cols in mutationMerged_dict if any(str(cols).startswith(val) for val in unique_values)]
        #[cols for cols in mutationMerged_dict for col in list(set(chain(*muts)))+names if col == cols]
        mutationMerged_dict = mutationMerged_dict.loc[:,['PATIENT_ID','SAMPLE_ID']].join(mutationMerged_dict.loc[:,getCols].fillna(0).astype(int))
        # Combine all data:
        patient_sample_data = pd.merge(all_clinical_data, mutationMerged_dict.merge(all_mut_data.rename(columns={'TUMOR_SAMPLE_BARCODE': 'SAMPLE_ID'}), 'left') , on=['PATIENT_ID','STUDY_NAME'], how='left')
        patient_sample_data = patient_sample_data.drop(['HUGO_SYMBOL','HGVSP','MUT_HGVSP'], axis=1)
        return patient_sample_data

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


