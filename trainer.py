import re
import torch
import argparse
import numpy as np
from utils_ensb import softmax
from itertools import combinations
from models.ensemble import Backwarder, Forwarder
from utils.trainer import free_params, frozen_params
from utils.eval import adjmat_to_cpdag_obj, cpdag_to_dag, dag_to_skeleton, write_result, evaluate, one_hot_edge_label

'''
Skeleton: (0) no edge, (1) i - j
Graph: (0) no edge, (1) i -> j ,  (2) j -> i
'''

class LossObj(torch.nn.Module):
    def __init__(self, eta):
        super(LossObj, self).__init__()
        self.eta = eta 
        self.loss_fn = torch.nn.BCELoss()
        self.cost_fn = torch.nn.MSELoss()
        
    def forward(self,output, yb):
        mprobs, probs, lmpr, lpr, _ = output
        bs = probs.shape[0]
        pr = torch.softmax(lpr, dim = 1)
        mpr = torch.softmax(lmpr, dim = 1)
        div = mpr * (lmpr - lpr) + pr * (lpr - lmpr)
        div = div.sum() / bs
        loss = self.loss_fn(probs, yb) + self.loss_fn(mprobs, yb)
        loss = loss + self.eta * div 
        return loss

def construct_model(config, device, devices):
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Embedding):
            torch.nn.init.xavier_normal_(m.weight)

    optim = torch.optim.Adam
    fmodel = Forwarder(config)
    fmodel = torch.nn.DataParallel(fmodel, device_ids=devices)
    fopt = optim(fmodel.parameters(), lr=config.lr)
    fmodel.to(device)

    
    bmodel = Backwarder(config)
    bmodel.apply(init_weights)
    bmodel = torch.nn.DataParallel(bmodel, device_ids=devices)
    bopt = optim(bmodel.parameters(), lr=config.lr)
    bmodel.to(device)

    return fmodel, bmodel, fopt, bopt

def train_epoch(data, loader, bmodel, fmodel, bopt, fopt, loss_fn, device):
    x, t, y, _, _ = data

    free_params(bmodel)
    frozen_params(fmodel)

    for indices in loader:
        xb, tb, yb = x[indices,], t[indices,], y[indices,]
        xb = xb.to(device)
        tb = tb.to(device)
        yb = yb.to(device)

        lpr = bmodel(yb, tb)
        out = fmodel(xb, tb, lpr)
        loss = loss_fn(out, yb)

        bopt.zero_grad()
        loss.backward()
        bopt.step()

    free_params(fmodel)
    frozen_params(bmodel)      

    for indices in loader:
        xb, tb, yb = x[indices,], t[indices,], y[indices,]
        xb = xb.to(device)
        tb = tb.to(device)
        yb = yb.to(device)
       
        lpr = bmodel(yb, tb)
        out = fmodel(xb, tb, lpr)
        loss = loss_fn(out, yb)
        
        fopt.zero_grad()
        loss.backward()
        fopt.step()
    return loss


class Trainer:
    
    def __init__(self, dataset, methods, sem_type, graph_dict):
        
        self.methods = methods
        self.sem_type = sem_type
        self.ngraph = max([len(graph_dict[mth]) for mth in graph_dict])
        self.num_nodes = dataset.num_nodes

        self.true_dag = dataset.B_bin
        self.true_skeleton = dag_to_skeleton(dataset.B_bin)
        self.true_cpdag = adjmat_to_cpdag_obj(dataset.B_bin, is_dag=True)

        self.features = list(combinations(range(self.num_nodes), 2))
    
        self.load_expert_votes(graph_dict) 
        self.load_config()
        self.expert_lookup = {expt: self.methods.index(re.findall(r"\D+",expt)[0]) for expt in self.experts}
        self.num_profiles = 2000

    def load_config(self):
        self.config = argparse.Namespace(
            lr=0.001,
            unit=None,
            num_layer=5, 
            embed_dim=64,
            vocab_size=None,
            time_scale_factor=100.,
            num_node=self.num_nodes,
            num_method=len(self.methods),
            discrete_dim=len(self.features)
    )


    def parse_dag(self, adj_matrix):
        graph_data, dir_data, skt_data = [], [], []
        for i,j in self.features: 
            if adj_matrix[i,j] == 1 and adj_matrix[j,i] == 0: # i -> j
                graph_data.append([1, -1, -1])
                dir_data.append([0, -1])
                skt_data.append(1)
            elif adj_matrix[i,j] == 0 and adj_matrix[j,i] == 1: # j -> i
                graph_data.append([2, -1, -1])
                dir_data.append([1, -1])
                skt_data.append(1)
            else: 
                graph_data.append([0, -1, -1])
                dir_data.append([-1, -1])
                skt_data.append(0)
        return graph_data, dir_data, skt_data

    def parse_cpdag(self, adj_matrix):
        graph_data, dir_data, skt_data = [], [], []
        for i,j in self.features: 
            if adj_matrix[j,i] == 1 and adj_matrix[i,j] == -1:    # i -> j
                graph_data.append([1, -1, -1])
                dir_data.append([0, -1])
                skt_data.append(1)
            elif adj_matrix[i,j] == 1 and adj_matrix[j,i] == -1:  # i <- j
                graph_data.append([2, -1, -1])
                dir_data.append([1, -1]) 
                skt_data.append(1)
            elif adj_matrix[i,j] == -1 and adj_matrix[j,i] == -1: # i - j
                graph_data.append([1, 2, -1])
                dir_data.append([0, 1])
                skt_data.append(1)
            else: # no edge
                graph_data.append([0, -1, -1])
                dir_data.append([-1, -1])
                skt_data.append(0)
        return graph_data, dir_data, skt_data

    def load_expert_votes(self, graph_dict):
        self.expert_votes = {'skeleton': {}, 'graph': {}, 'direction': {}}
        self.expert_dags = {}
        self.expert_cpdags = {} # cpdag objs
        self.experts = []
        for mth in self.methods:
            graph_list = graph_dict[mth]
            for i, arr in enumerate(graph_list):
                expt = mth + str(i)
                if 'GES' in mth or 'PC' in mth: 
                    cpmat = arr 
                    dagmat = cpdag_to_dag(cpmat)
                    graph, direction, skeleton = self.parse_cpdag(cpmat)
                    self.expert_cpdags[expt] = adjmat_to_cpdag_obj(cpmat, is_dag=False)
                else:
                    dagmat = arr
                    graph, direction, skeleton = self.parse_dag(dagmat)
                    self.expert_cpdags[expt] = adjmat_to_cpdag_obj(dagmat, is_dag=True)
                
                self.expert_dags[expt] = dagmat
                self.expert_votes['graph'][expt] = graph
                self.expert_votes['skeleton'][expt] = skeleton
                self.expert_votes['direction'][expt] = direction
                self.experts.append(expt)
            
            
    def vote_to_cpmat(self, feature_score_dict, unit):
        B_est = np.zeros((self.num_nodes,self.num_nodes))
        for (i,j) in self.features:
            value = feature_score_dict[(i,j)].argmax()
            if unit == 'skeleton' and value == 1:
                B_est[i,j] = B_est[j,i] = 1
            if (unit == 'graph' and value == 1) or (unit == 'direction' and value == 0): 
                B_est[i,j], B_est[j,i] = -1, 1
            if (unit == 'graph' and value == 2) or (unit == 'direction' and value == 1): 
                B_est[i,j], B_est[j,i] = 1, -1   
        return B_est

    
    def generate_input(self, unit):
        '''
        batch size: num feats x num expts 
        x : method index
        t : feature index
        y : one-hot label
        '''
        
        steps, x, y, meta = [], [], [], []
        for i, ft in enumerate(self.features):
            for expt in self.experts:
                lab = self.expert_votes[unit][expt][i]
                vote = self.expert_votes[unit][expt]

                if unit == 'skeleton':
                    one_hot = torch.eye(2)[lab]

                elif unit == 'graph':
                    vote = np.array(vote)
                    vote = np.where(vote < 0, 3, vote).tolist()
                    one_hot = one_hot_edge_label(lab)   
                
                x.append(self.expert_lookup[expt])
                steps.append(i)  
                y.append(one_hot)     
                meta.append((ft, expt))
              
                              
        t = torch.LongTensor(steps)
        x = torch.LongTensor(x)
        binary_votes = np.stack(y, axis=0).astype("float32")
        y = torch.from_numpy(binary_votes)
    
        return x, t, y, meta, binary_votes
    
    def evaluate_experts(self, gtype, method_list = None, write_output=False):
        if method_list is None: method_list = self.methods 
        if gtype == 'CPDAG':
            graph_dict = self.expert_cpdags  
            truth = self.true_cpdag
        else: 
            graph_dict = self.expert_dags
            truth = self.true_dag

        reports = {}
        for mth in method_list: 
            results = []
            for i in range(self.ngraph):
                try:
                    if gtype == 'SKT': 
                        if 'PC' in mth or 'GES' in mth: 
                            est = - np.abs(self.expert_cpdags[mth + str(i)].graph)
                        else:
                            dag = graph_dict[mth + str(i)]
                            est = - dag_to_skeleton(dag)
                        truth = - self.true_skeleton
                    else: 
                        est = graph_dict[mth + str(i)]
                    result_dict = evaluate(est , truth, gtype)  
                    results.append(result_dict.metrics) 
                except ZeroDivisionError: 
                    pass
            metrics = {k : np.nanmean([item[k] for item in results]) for k in result_dict.metrics}
            result_dict.metrics = metrics
            reports[mth] = metrics
            if write_output: write_result(result_dict, mth + '-' + gtype, f'output/{self.sem_type}/{gtype}.txt')    
        return reports   

   
        
    def test(self, B_est, unit, view = True):
        if unit == 'skeleton':
            res = evaluate(-B_est, -self.true_skeleton, gtype = 'SKT', display = view)
        else:
            est_cpdag =  adjmat_to_cpdag_obj(B_est, is_dag=False)
            res = evaluate(est_cpdag, self.true_cpdag, gtype = 'CPDAG', display = view)
        return res
          
    def aggregate(self, meta, score_array):
        collector = [{ft: [None] * len(self.methods) for ft in self.features} for _ in range(self.ngraph) ]
        for i, (ft, expt) in enumerate(meta): 
            profile = int(re.findall(r"\d+",expt)[0])
            loc = self.expert_lookup[expt]
            if ft in self.features:
                collector[profile][ft][loc] = score_array[i, ]
            
        for i in range(self.ngraph):
            for ft in self.features: 
                collector[i][ft] = np.stack(collector[i][ft], axis=1)
        return collector

    def predict(self, unit, meta, votes, logT, logPr, weighted_by = 'map', skeleton = None):
        
        num_class = votes.shape[1]
        if weighted_by != 'mjr': 
            if weighted_by == 'tmap': 
                logPr = logPr /  len(self.experts)
            else:
                T = softmax(logT)
                Pr = softmax(logPr)
                votes = np.matmul(T, np.expand_dims(Pr, axis=2)).reshape(-1, num_class) * votes
                
        
        # else: majority voting
        collector = self.aggregate(meta, votes)
        dicts = []
        for i in range(self.ngraph):
            output = {}
            for ft, scr in collector[i].items():  
                output[ft] = np.sum(scr, axis=1)
            B_est = self.vote_to_cpmat(output, unit)
            if skeleton is not None: B_est = B_est * skeleton
            res = self.test(B_est, unit, view = False)
            dicts.append(res.metrics)
        
        res.metrics = {k: np.nansum([dic[k] for dic in dicts]) / self.ngraph for k in dicts[0]}
        return res
 