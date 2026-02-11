import sys
import torch
from tqdm import tqdm
from trainer import *
from utils_ensb import *
from utils.io import write_pickle
from torch.utils.data import DataLoader


unit = "graph"
ngraph = int(sys.argv[1])
sem_type = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

methods, dataset, _ = get_setting(sem_type)
    
loss_fn = LossObj(0.001) 
num_epochs = 500
batch_size = 10000


logTs, logPrs = [], []
rpts = 30
for r in range(rpts):

    # Load trainer
    graph_dict, loc_dict = load_graph_dict(sem_type, methods, ngraph = ngraph)
    trainer = Trainer(dataset, methods, sem_type, graph_dict = graph_dict)
    data = trainer.generate_input(unit)
    x, t, y, meta, votes = data
    trainer.config.vocab_size = votes.shape[1]
    trainer.config.unit = unit

    batches = list(range(x.shape[0]))
    loader = DataLoader(batches, batch_size=batch_size, shuffle=True)

    # Train model
    pbar = tqdm(range(num_epochs))
    fmodel, bmodel, fopt, bopt = construct_model(trainer.config, device, devices=[0])
    for epoch in pbar: 
        loss = train_epoch(data, loader, bmodel, fmodel, bopt, fopt, loss_fn, device)
        pbar.set_description(f'step {r}: {loss.item():.4f}')
        
    fmodel.eval()
        

    _, _, logPr, _, logT = fmodel(x, t, None)

    logT = logT.detach().cpu().numpy()
    logPr = logPr.detach().cpu().numpy()
    logTs.append(logT)
    logPrs.append(logPr)


logTa = np.mean(logTs,   axis=0)
logPra = np.mean(logPrs, axis=0)
output = (logTa, logPra)

save_path = f'output/{sem_type}/{unit}-{ngraph}.pkl'
write_pickle(output, save_path)



