import sys
import random
import numpy as np
from tqdm import tqdm
from baseline import *
from trainer import Trainer
from utils.eval import evaluate, cpdag_to_dag, get_edge_label, one_hot_edge_label
from utils.io import load_pickle
from utils_ensb import get_setting, load_graph_dict, softmax


'''
Multi-step ensembling to obtain a final graph 
+ Option 1: Inter-algo aggregation (one graph per expert) --> Intra-algo aggregation
+ Option 2: Intra-algo aggregation (a set of graphs from different voting profiles) --> Inter-algo aggregation

O1 - [MJR, RANK] - [MJR, RANK, BAYES]
O2 - [MJR, RANK, BAYES] - [MJR, RANK]
'''

unit = "graph"
ngraph = int(sys.argv[1])
sem_type = sys.argv[2]
logTa,  logPra = load_pickle(f'output/{sem_type}/{unit}-{ngraph}.pkl')


methods, dataset = get_setting(sem_type)
graph_dict, _ = load_graph_dict(sem_type, methods, ngraph = None, loc_dict = None)
trainer = Trainer(dataset, methods, sem_type, graph_dict = graph_dict)



       
meta = []
for i, ft in enumerate(trainer.features):
    for algo in methods: 
        lis = [(ft, algo + str(i))for i in range(50)]
        meta.extend(lis)
            
# Retrieve weights uniquely by feature and method
names = [algo + '0' for algo in methods]
locs = [i for i, ( _, name ) in enumerate(meta) if name in names]
logT = logTa[locs, ]
logPr = logPra[locs, ]
meta = [meta[l] for l in locs]

T = softmax(logT)
Pr = softmax(logPr)

def adjmat_to_vote(adj_matrix):
    vote, _, _ = trainer.parse_dag(adj_matrix)
    vote = [one_hot_edge_label(label) for label in vote]
    return vote

def bayes_voting(profile):
    """
    dict: key = method name + '0', value = adj matrix
    """


    collector = {k : {} for k in meta}
    # Load new votes and organize votes by label 

    for algo, adj_matrix in profile.items():
        single_vote = adjmat_to_vote(adj_matrix)
        for i, ft in enumerate(trainer.features):
            collector[(ft, algo)] = single_vote[i]
    
    votes = [collector[key] for key in meta]
    votes = np.stack(votes, axis=0).astype("float32")

    # change votes
    votes = np.matmul(T, np.expand_dims(Pr, axis=2)).reshape(-1, 3) * votes

    output = {ft: 0 for ft in trainer.features}
    for i, (ft, expt) in enumerate(meta):
        output[ft] += votes[i,]

    est = trainer.vote_to_cpmat(output, "graph") # this follows cpdag syntax
    est = cpdag_to_dag(est)
    return est
                


def get_mode(lis):
    return max(set(lis), key=lis.count)


def majority_voting(graph_list):
    dic = {e: [] for e in trainer.features}
    for B_est in graph_list:
        for i,j in trainer.features:
            y = get_edge_label(B_est, i, j)
            dic[(i,j)].append(y)
    
    est = np.zeros_like(B_est)
    for i,j in trainer.features: 
        value = get_mode(dic[(i,j)])
        if value == 1: 
            est[i,j], est[j,i] = 1, 0
        elif value == 2:
            est[i,j], est[j,i] = 0, 1
        elif value == 3:
            est[i,j], est[j,i] = 1, 1
    return est

def rank_voting(graph_list):
    graphs = np.stack(graph_list, axis=2)
    est, _, _ = update_center_greedy(graphs, p=1, q=0)
    return est



def aggregate_inter_algo_graph(base_algo, by):
    graph_list = []

    for expt, adj_matrix in trainer.expert_dags.items():
        if base_algo in expt: 
            adj_matrix = np.asarray(adj_matrix)
            graph_list.append(adj_matrix)

    if by == 'majority': 
        return majority_voting(graph_list)
    else: 
        return rank_voting(graph_list)


def get_voting_collection(num_profiles):
    profiles = []
    for _ in tqdm(range(num_profiles)):
        profile = {}
        for mth in methods: 
            N = len(graph_dict[mth])
            adjmat = graph_dict[mth][random.choice(range(N))]
            if 'PC' in mth or 'GES' in mth: 
                adjmat = cpdag_to_dag(adjmat)
            adjmat = np.asarray(adjmat)
            profile[mth + '0'] = adjmat
        profiles.append(profile)
    return profiles


def aggregate_intra_algo_graph(profile_list, ensb_algo, by):
    '''
    ensb_algo: bayes, majority, rank
    ''' 
    graph_list = []
    
    for profile in profile_list: 

        grlis = [adjmat for _, adjmat in profile.items()]
        
        if ensb_algo == 'bayes':
            A = bayes_voting(profile)
        elif ensb_algo == 'rank':
            A = rank_voting(grlis)
        else: 
            A = majority_voting(grlis)
        graph_list.append(A)
    
    
    if by == 'majority': 
        return majority_voting(graph_list)
    else: 
        return rank_voting(graph_list)

def run_option1(step1, step2): 
    # Output a DAG-format
    assert step1 in ('majority', 'rank')
    profile, graph_list = {}, []
    for base_algo in methods: 
        est = aggregate_inter_algo_graph(base_algo, by = step1)
        est = np.asarray(est)
        profile[base_algo + '0'] = est
        graph_list.append(est)

    if step2 == 'bayes': 
        est = bayes_voting(profile)
    elif step2 == 'rank':
        est = rank_voting(graph_list) 
    else:
        est = majority_voting(graph_list) 
    return est
    


def run_option2(profile_list, step1, step2): 
    assert step2 in ('majority', 'rank')
    # Output a DAG-format
    est = aggregate_intra_algo_graph(profile_list, step1, step2)
    return est


lis1 = ('bayes', 'majority', 'rank')
lis2 = ('majority', 'rank')

print('=' * 30, 'Option 1', '=' * 30)
for step1 in lis2: 
    for step2 in lis1: 
        est = run_option1(step1, step2)
        res = evaluate(est , dataset.B_bin, "DAG")   
        print(f'SHD = {res.metrics["shd"]} , F1 = {res.metrics["F1"]:.3f} | O1-{step1}-{step2} ')

print('Get voting collections')
profile_list = get_voting_collection(2000)
print('=' * 30, 'Option 2', '=' * 30)
for step1 in lis1: 
    for step2 in lis2: 
        est = run_option2(profile_list, step1, step2)
        res = evaluate(est , dataset.B_bin, "DAG")   
        print(f'SHD = {res.metrics["shd"]} , F1 = {res.metrics["F1"]:.3f} | O2-{step1}-{step2}')

