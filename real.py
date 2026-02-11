import os
import numpy as np
import pandas as pd
import networkx as nx
from utils.data import produce_NA
from cdt.data import load_dataset
from utils.io import load_pickle, write_pickle

class RealDataset:
    def __init__(self, root, config_code, sem_type, miss_config):
        self.data_path = f'./{root}/{config_code}.pickle'

        if miss_config is not None:
            miss_type, miss_percent = miss_config
        
        self.X = None
        
        
        if os.path.isfile(self.data_path):
            self.B_bin, self.X_true, self.X = load_pickle(self.data_path)
        else:
            print('Generating and Saving data ...')
            
            
            if 'sachs' in sem_type:
                s_data, s_graph = load_dataset('sachs') 
                if sem_type == 'sachsraw':           
                    self.X_true = s_data.to_numpy()
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    self.X_true = scaler.fit_transform(s_data)
                    
                self.B_bin = nx.adjacency_matrix(s_graph).todense()

            
            elif sem_type == 'artic':
                self.X_true = pd.read_csv('dataset/storage/artic_data.csv').values
                self.B_bin = pd.read_csv('dataset/storage/artic_graph.csv', index_col = 0).values
            
            elif sem_type == 'sangiovese':
                df = pd.read_csv('dataset/storage/sangiovese.csv', index_col=0)
                self.X_true = df.values 
                colnames = list(df.columns)
                from utils.io import load_json 
                B_bin = np.zeros((len(colnames), len(colnames)))
                scm = load_json('dataset/storage/sangiovese.json')
                for j, col in enumerate(colnames): 
                    for pa in scm[col]['parent']: 
                        if pa != 'Treatment': 
                            i = colnames.index(pa)
                            B_bin[i,j] = 1
                self.B_bin = B_bin

            else: 

                from pgmpy.utils import get_example_model
                model = get_example_model(sem_type)
                file_path = f'dataset/{sem_type}.csv'
                df = pd.read_csv(file_path)
                colnames = [name for name in df.columns]
                B_bin = np.zeros((df.shape[1], df.shape[1]))
                self.X_true = df.values
                
                for vi, vj in model.edges():
                    i,j = colnames.index(vi), colnames.index(vj)
                    B_bin[i,j] = 1
                self.B_bin = B_bin
                del model      

            if miss_config is not None and self.X is None: 
                self.X = produce_NA(self.X_true, miss_percent, mecha=miss_type, opt="logistic", p_obs=0.3, q=0.3)
            
            package = (self.B_bin, self.X_true, self.X)
            write_pickle(package, self.data_path)
        
# https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2021.642182/full
# https://github.com/big-data-lab-umbc/cybertraining/tree/master/year-3-projects/team-6
# https://zenodo.org/records/838571