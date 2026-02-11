from real import RealDataset
from synthetic import SyntheticDataset
import os



def get_config(config_id, graph_type, sem_type, equal_variances, missing):
    assert graph_type in ('ER', 'SF', 'REAL'), 'ER/SF/REAL graph only'
    config = {
            'num_obs': 5000,
            # 'num_vars': 50,
            'graph_type': graph_type,
            'degree': 4, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.3,
            "sem_type": sem_type.lower(),
            "ev": equal_variances
        }
    # Customized config id here
    config['code'] = f'{sem_type}-{graph_type}'
    if missing is None:
        config['miss_percent'] = 0.0
    
    if graph_type == 'REAL':
        return config

    i = int(str(config_id)[-1])
    if 'GP' in sem_type:
        num_vars_ = [5, 10, 15, 20, 25, 30]
    else: 
        num_vars_ = [10, 20, 40, 60, 80, 100]
    config['num_vars'] = num_vars_[i - 1]
    
    if config_id > 10 and config_id < 20:
        config['noise_type'] = 'gumbel'
    elif config_id > 20 and config_id < 30:
        config['noise_type'] = 'laplace'
    elif config_id > 30 and config_id < 40:
        config['noise_type'] = 'uniform'

    

    abbr = 'EV' if equal_variances else 'NV'
    config['code'] = config['code'] + f'-{abbr}{config_id}'
    
    return config


def get_data(root, config_id, graph_type, sem_type, equal_variances, missing):

    config = get_config(config_id, graph_type, sem_type, equal_variances, missing)
    
    if graph_type == 'REAL': 
        if config['miss_percent'] == 0.0:
            miss_config = None 
        else: 
            miss_config = (config['miss_type'], config['miss_percent'])
        dataset = RealDataset(root = root, config_code = config['code'],
                                sem_type = sem_type, miss_config = miss_config)
    else:
        
        dataset = SyntheticDataset(root = root, n = config['num_obs'], d = config['num_vars'], 
                            config_code = config['code'],
                            graph_type = config['graph_type'], 
                            degree = config['degree'], 
                            noise_type = config['noise_type'],
                            miss_type = config['miss_type'], 
                            miss_percent = config['miss_percent'], 
                            sem_type = config['sem_type'],
                            equal_variances = config['ev'],
                            )

    return dataset, config


def get_output_path(sem_type, method, graph_type, ev):
    variance_type = 'EV' if ev else 'NV'
    os.makedirs(f'output/{sem_type}/', exist_ok=True)
    if graph_type == 'REAL': 
        output_path = f'output/{sem_type}/{method}'
    else:
        output_path = f'output/{sem_type}/{method}-{graph_type}-{variance_type}'
    return output_path 