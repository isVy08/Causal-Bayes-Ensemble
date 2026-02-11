import pickle
import torch
import numpy as np
import json

def load_txt(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def write_txt(data, savedir, mode='w'):
    f = open(savedir, mode)
    for text in data:
        f.write(text+'\n')
    f.close()


def load_pickle(datadir):
    file = open(datadir, 'rb')
    data = pickle.load(file)
    return data


def write_pickle(data, savedir):
    file = open(savedir, 'wb')
    pickle.dump(data, file)
    file.close()


def load_model(model, optimizer, scheduler, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'prev_loss' in checkpoint.keys():
        return checkpoint['prev_loss']

# frozen_params removed - use utils.trainer.frozen_params instead

def load_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the JSON data into a dictionary
        data = json.load(file)
    # Now 'data' contains the dictionary representation of the JSON file
    return data

def write_json(data, savedir):
    with open(savedir, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
