import torch
import sys, os
import numpy as np
from utils_ensb import get_setting
from utils.eval import postprocess
from data_generator import get_output_path
from utils.io import load_pickle, write_pickle

'''
Generate a list of graph predictions from an algorithmic expert
'''

sem_type = sys.argv[1]  
method = sys.argv[2]

root = 'dataset/'
_, dataset, graph_type = get_setting(sem_type)
saved_path = get_output_path(sem_type, method, graph_type, False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X = dataset.X_true


N, D = X.shape
if os.path.isfile(saved_path + '.pkl'):
    predictions = load_pickle(saved_path + '.pkl')
    # sample with replacement
    rands = np.random.choice(N, N, replace=True)
    X = X[rands, :]

else: 
    predictions = []


if method == 'DAGMA':
    
    from models.dagma import DagmaMLP, DagmaNonlinear
    X = torch.from_numpy(X).to(device)
    X = X.float()
    eq_model = DagmaMLP(dims=[D, D, 1], bias=True)
    eq_model.to(device)
    model = DagmaNonlinear(eq_model)
    W_est = model.fit(X, lambda1=0.02, lambda2=0.005, warm_iter=5e4, max_iter=8e4)
    _, B_est = postprocess(W_est, graph_thres = 0.3)

elif method == 'NOTEARS':
     
    X = X.astype("float32")
    from models.notears import NotearsMLP, notears_nonlinear
    model = NotearsMLP(dims=[D, D, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01) 
    _, B_est = postprocess(W_est, graph_thres = 0.3)
    
elif 'GES' in method: 
    from causallearn.search.ScoreBased.GES import ges
    if 'BIC' in method:
        Record = ges(X, 'local_score_BIC')
    elif 'MRG' in method:
        Record = ges(X, 'local_score_marginal_multi')
    B_est = Record['G'].graph


elif 'PC' in method:
 
    from causalai.models.tabular.pc import PCSingle, PC
    from causalai.data.tabular import TabularData

   
    if method == 'PC-KCI':
        from causalai.models.common.CI_tests.kci import KCI
        CI_test = KCI(chunk_size=100)
    else:
        from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
        CI_test = PartialCorrelation()
    
    var_names = [str(i) for i in range(D)]
    data_obj = TabularData(X, var_names=var_names)
    pvalue = 0.01
    pc = PC(data=data_obj, prior_knowledge=None, CI_test=CI_test,use_multiprocessing=False)
    result = pc.run(pvalue_thres=pvalue, max_condition_set_size=D-2)
    B_est = np.zeros_like(dataset.B_bin)
    for key in result.keys():
        parents = result[key]['parents']
        for pa in parents :
            B_est[int(key), int(pa)] = -1
            B_est[int(pa), int(key)] = 1
        for node in pc.skeleton[key]: 
            if node not in parents: 
                B_est[int(key), int(node)] = B_est[int(node), int(key)] = -1

elif method == 'LiNGAM':
    from causallearn.search.FCMBased import lingam
    model = lingam.ICALiNGAM(1234, 10000)
    model.fit(X)
    B_est = (model.adjacency_matrix_ > 0).astype('int')


# df = pd.DataFrame(B_est)
predictions.append(B_est)
write_pickle(predictions, saved_path + '.pkl')
