import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.trainer import permute_rows_soft
from utils.sampler import Sample_Categorical



    
class Forwarder(nn.Module):
    def __init__(self, config):
        super(Forwarder, self).__init__()    
        self.ddim = config.discrete_dim
        self.vcb = config.vocab_size
        self.num_method = config.num_method
      
        self.trans_logits = nn.Parameter(torch.empty(self.ddim, self.num_method, self.vcb , self.vcb))
        self.prior_logits = nn.Parameter(torch.empty(self.ddim, self.vcb))

        nn.init.normal_(self.trans_logits)
        nn.init.normal_(self.prior_logits) 

        self.sampler = Sample_Categorical(2.0)

        

    
    def forward(self, x, t, lpr):

        bs = x.shape[0]
        
        toh = torch.eye(self.ddim, device = x.device)[t]
        xoh = torch.eye(self.num_method, device = x.device)[x].unsqueeze(dim=1)

        # lmpr = self.prior_logits[t, :]
        # trans_matrix = self.trans_logits[t, x, :, :]

        lmpr = torch.matmul(toh, self.prior_logits)
        trans_matrix = self.trans_logits.view(self.ddim, -1)
        trans_matrix = torch.matmul(toh, trans_matrix).view(bs, self.num_method, -1)
        trans_matrix = torch.matmul(xoh, trans_matrix).view(bs, self.vcb, self.vcb)

        trans_matrix = permute_rows_soft(trans_matrix, None, tau=0.2)
        
        T = torch.softmax(trans_matrix, dim = 2)
        
        # Sample
        probs = None 
        if lpr is not None:
            prior = torch.softmax(lpr, dim=1)
            prior = self.sampler(prior)
            probs = torch.matmul(prior.unsqueeze(dim=1), T).squeeze(dim=1)
        
        mprior = torch.softmax(lmpr, dim=1)
        mprobs = torch.matmul(mprior.unsqueeze(dim=1), T).squeeze(dim=1)
        
        
        
        return mprobs, probs, lmpr, lpr, trans_matrix


class Backwarder(nn.Module):
    def __init__(self, config):
        super(Backwarder, self).__init__()
        self.time_scale_factor = config.time_scale_factor
        self.embed_dim = config.embed_dim
        vcb = config.vocab_size

        self.hidden_size = self.embed_dim * vcb
        self.input_layer = nn.Linear(vcb, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, config.vocab_size)
        

    def forward(self, y, t):
        '''
        y : batch size, seq length (num class)
        t : batch size, (feature index)
        returns prior : batch size, num class
        '''
        y = self.input_layer(y)
        y = F.relu(y)
        logits = self.output_layer(y) 
        return logits
