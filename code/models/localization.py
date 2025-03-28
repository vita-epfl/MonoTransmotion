import torch
import torch.nn as nn
import numpy as np

def create_loc_model(config):

    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]
    if config["MODEL"]["type"] == "transmotion":
        model = TransMotion(tok_dim=seq_len,
            nhid=nhid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            output_scale=config["MODEL"]["output_scale"],
            num_tokens=token_num,
            device=config["DEVICE"]
        ).to(config["DEVICE"]).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model


class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=10, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)
        self.seq_len = seq_len
        

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        seq_len = self.seq_len
        half = x.size(2)//2 ## 124
        x[:,:,:] = x[:,:,:] + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(0)

        return self.dropout(x)


class TransMotion(nn.Module):
    def __init__(self, tok_dim=10, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, dropout=0.1, activation='relu', output_scale=1, num_tokens=47, device='cuda:0'):

        super(TransMotion, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.joints_pose = 22
        self.obs_and_pred = tok_dim
        self.device = device
        
        self.fc_in_traj = nn.Linear(34,nhid)
        self.fc_out_traj = nn.Linear(nhid, 34)

        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=self.seq_len, device=device) 


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
                                                   
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)
       
        # simplified projection head
        self.regression_head = nn.Sequential(
            nn.Linear(nhid, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 9)  # Output layer for regression
        )
    
    def forward(self, tgt, padding_mask):
   
        B, in_F, NJ, K = tgt.shape 
        F = self.obs_and_pred 
        J = self.token_num

        out_F = F - in_F
        N = NJ // J
        
        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)  
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)
        ## add mask
        mask_ratio_traj = 0.0 # 0.1 for training

        tgt_traj = tgt[:,:,:,0,:2].to(self.device) 
        traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask
        
        # reshape (16,10,17,2) => (16, 10, 34)
        tgt_traj = tgt_traj.reshape(B,F,N*2)
        ############
        # Transformer
        ###########


        tgt_traj = self.fc_in_traj(tgt_traj) # 16, 10, 128
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) # 16, 10, dim
  
        tgt_padding_mask_local = padding_mask[:,:self.seq_len]
        tgt_padding_mask_local[:,in_F:] = True

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid) 
        tgt = tgt_traj

        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local) #!!!

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt # 10, 16, 128

        # 10, 16, 128 => 16, 10, 128
        out_local = out_local.transpose(0,1)
        # 16, 10, 128 => 160, 128
        out_local = out_local.reshape(out_local.size(0)*out_local.size(1), out_local.size(2))
        
        out_local = self.regression_head(out_local)

        return out_local