import sys
import torch
import string
import numpy as np
from utils_ensb import *
from trainer import *
from torch.utils.data import DataLoader

# seed = 1234
# torch.manual_seed(seed)
# np.random.seed(seed)


def learn_params(trainer, unit, num_epochs = 1000, batch_size = 10000):
    
    data = trainer.generate_input(unit)
    x, t, y, meta, votes = data
    loss_fn = LossObj(0.001)
    print('Training begins for', unit, 'on', x.shape[0], 'samples!')
    trainer.config.vocab_size = votes.shape[1]
    trainer.config.unit = unit
    fmodel, bmodel, fopt, bopt = construct_model(trainer.config, device, devices=[0])

    batches = list(range(x.shape[0]))
    loader = DataLoader(batches, batch_size=batch_size, shuffle=True)
    
    pbar = range(num_epochs)
    
    for _ in pbar: 
        _ = train_epoch(data, loader,  bmodel, fmodel, bopt, fopt, loss_fn, device)
    fmodel.eval()
    _, _, logPr, _, logT = fmodel(x, t, None)
    logT = logT.detach().cpu().numpy() 
    logPr = logPr.detach().cpu().numpy() 

    gr_data = (meta, votes, logT, logPr)
    return gr_data

'''
D : [5, 10, 15, 20]
M : [5, 7, 9, 11]
'''

sem_type = 'SIM'
num_nodes = int(sys.argv[1])
num_methods = int(sys.argv[2])
ngraph = int(sys.argv[3]) # num graphs
stg = sys.argv[4]

print(f'SETTING: {num_nodes} - {num_methods} - {stg}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
methods = list(string.ascii_uppercase[:num_methods])
if stg == 'st': # strong (0.5, 1.0)
    B_bin, T, pr, graph_dict  = simulate_data(methods, num_nodes, ngraph = ngraph)
elif stg == 'md': # medium
    B_bin, T, pr, graph_dict  = simulate_data(methods, num_nodes, ngraph = ngraph, lb = 0.4, ub = 0.6) 
else: # weak
    B_bin, T, pr, graph_dict  = simulate_data(methods, num_nodes, ngraph = ngraph, lb = 0.3, ub = 0.5)
dataset = DataObj(B_bin)
trainer = Trainer(dataset, methods, sem_type, graph_dict = graph_dict)



# Test truth
x, t, _, meta, votes = trainer.generate_input('graph')
t = t.numpy()
logT = np.log(T)[t, x, ]
logPr = np.log(pr)[t,]
gr_data = (meta, votes, logT, logPr)
tmap_res = trainer.predict("graph", meta, votes, logT, logPr, weighted_by = 'tmap')
# Performance reports
sk_data = None
    
reports = trainer.evaluate_experts('CPDAG', write_output = False)

# sk_data = learn_params(trainer, 'skeleton')
meta, votes, logT, logPr = learn_params(trainer, 'graph')


mjr_res = trainer.predict("graph", meta, votes, logT, logPr, weighted_by = 'mjr')
map_res = trainer.predict("graph", meta, votes, logT, logPr, weighted_by = 'map')
res_dict = {'MAJRT': mjr_res, 'E-MAP': map_res, 'T-MAP': tmap_res}
comparison(res_dict, reports)

'''
Estimation Error
'''
est_T = softmax(logT)
est_pr = softmax(logPr)

T = T[t, x, ]
pr = pr[t,]

comp = np.diagonal(T, axis1=1, axis2=2)
est_comp = np.diagonal(est_T, axis1=1, axis2=2)

cerr = np.square((T - est_T)).mean(axis=0)
perr = np.square((pr - est_pr)).mean()

div = np.diagonal(est_T - T, axis1=1, axis2=2)

print('Competence error:', cerr.mean(axis=1))
print('Prior error:', perr)

value = comp.mean(axis=0).round(4)
est_value = est_comp.mean(axis=0).round(4)
print('Truth T:', value, value.argsort()[::-1])
print('Estim T:', est_value, est_value.argsort()[::-1])

value = pr.mean(axis=0).round(4)
est_value = est_pr.mean(axis=0).round(4)
print('Truth P:', value)
print('Estim P:', est_value)



