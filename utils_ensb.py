import torch
import numpy as np
from data_generator import get_data
from itertools import combinations
from utils.io import load_pickle




class DataObj:
    def __init__(self, B_bin):
        self.B_bin = B_bin
        self.num_nodes = B_bin.shape[0]


def load_graph_dict(sem_type, methods, ngraph = None, loc_dict = None):
    graph_dict = {}

    if loc_dict is None: 
        loc_dict = {mth: None for mth in methods}

    for mth in methods: 
        if sem_type == 'MLP':
            graph_list = load_pickle(f'output/MLP/{mth}-MLP-SF-NV.pkl')
        elif sem_type == 'GP':
            graph_list = load_pickle(f'output/GP/{mth}-GP-ER-NV.pkl')
        else:
            graph_list = load_pickle(f'output/{sem_type}/{mth}.pkl')
        
        if ngraph is None: ngraph = len(graph_list)
            
        N = len(graph_list)
        if loc_dict[mth] is None: 
            
            if N >= ngraph: 
                locs = np.random.choice(N, ngraph, replace=False)
            else: 
                locs = np.random.choice(N, ngraph, replace=True)
            
            loc_dict[mth] = locs
        else: 
            locs = loc_dict[mth]
        graph_list = [graph_list[i] for i in locs]
        graph_dict[mth] = graph_list
    return graph_dict, loc_dict

def get_setting(sem_type):
    methods = ['DAGMA','NOTEARS','SCORE','PC','GES-BIC','LiNGAM']
    if sem_type == 'MLP':
        config_id, graph_type = 23, 'SF'
    elif sem_type == 'GP':
        config_id, graph_type = 34, 'ER'
    else:
        config_id, graph_type = 1, 'REAL'
    
    dataset, _ = get_data('dataset', config_id, graph_type, sem_type, False, None)
    
    if sem_type in ('child','insurance','asia','earthquake'):
        methods = ['HCS','PC','GES']
    
    dataset.num_nodes = dataset.X_true.shape[1]
    dataset.X_true = None
    return methods, dataset, config_id, graph_type


def softmax(logarr):
    exp = np.exp(logarr)
    sumexp = exp.sum(axis=-1)
    probs = exp / np.expand_dims(sumexp, axis=-1)
    return probs 


def simulate_data(methods, num_nodes, ngraph, lb=0.5, ub=1.0):

    from utils.graph import fill_triangular
    import torch.distributions as td

    def to_graph(data):
        '''
        For each pair of nodes xᵢ and xⱼ where i < j, sample a three way categorical Cᵢⱼ.
        Cᵢⱼ = 1, represents the edge xᵢ -> xⱼ,
        Cᵢⱼ = 2, represents the edge xᵢ <- xⱼ,
        Cᵢⱼ = 0, there is no edge between these nodes.
        '''
        G = fill_triangular(data[..., 2], upper=False) + fill_triangular(data[..., 1], upper=True)
        return G

    num_methods = len(methods)
    features = list(combinations(range(num_nodes), 2))
    num_features = len(features)

    # Transition matrix size (#features, #methods, 3 x 3)
    N, D = num_features * num_methods, 3
    
    diag = torch.empty(N,D).uniform_(lb, ub)
    rest = torch.rand(N, D, 2)
    rest = rest / rest.sum(dim=2, keepdim=True)  
    rest = (rest * (1 - diag.unsqueeze(dim=2))).reshape(-1,)
    trans_matrix = torch.zeros(N, D, D)

    diag_indices = torch.arange(D)
    trans_matrix[torch.arange(N)[:, None], diag_indices, diag_indices] = diag
    non_diag_mask = ~torch.eye(D, dtype=bool)
    non_diag_mask = non_diag_mask.unsqueeze(0).expand(N, -1, -1)
    trans_matrix[non_diag_mask] = rest
    T = trans_matrix.view(num_features, num_methods, D, D)

    # Prior and true DAG
    p0 = torch.empty(num_features).uniform_(0.5, 1.0)  # edge existence at most 0.5 due to DAG
    rest = torch.rand(num_features, 2)
    rest = rest / rest.sum(dim=1, keepdim=True)  # Normalize
    rest = rest * (1 - p0).unsqueeze(1)          # Scale to 1 - p0
    pr = torch.cat([p0.unsqueeze(1), rest], dim=1).view(num_features, 1, 1, 3)

    Gdist = td.Independent(td.OneHotCategorical(probs=pr.squeeze(), validate_args=False), 1)
    data = Gdist.sample()
    B_bin = to_graph(data)

    # Generate noisy graphs
    oh = data.view(-1, 1, 1, 3)
    probs = torch.matmul(oh, T).squeeze(dim=2) # (#features, #methods, 3)
    graph_dict = {}
    for m, mth in enumerate(methods):
        _probs_ = probs[:, m, :]
        dist = td.Independent(td.OneHotCategorical(probs=_probs_, validate_args=False), 1)
        graph_list = []
        for _ in range(ngraph):
            data = dist.sample()
            B_est = to_graph(data)
            graph_list.append(B_est)
        graph_dict[mth] = graph_list

    pr = pr[:, 0, 0, :].numpy()
    T = T.numpy()
    return B_bin, T, pr, graph_dict




def comparison(res_dict, reports, metrics = ('shd','adjF1','arrowF1')): 

    for mtr in metrics:
        if reports is not None:
            arr = np.array([reports[mth][mtr] for mth in reports])
            arr = arr[~np.isnan(arr)]
            if arr.shape[0] == 0:
                minval = maxval = 0.
            else:
                minval, maxval = arr.min(), arr.max()    
            print(f'{mtr}: min = {minval:.4f}, max = {maxval:.4f}')
        for name, res in res_dict.items():
            try:
                print(f'{name}: {res.metrics[mtr]:.4f}')
            except:
                print(f'{name}: {res[mtr]:.4f}')
        print('=' * 30)