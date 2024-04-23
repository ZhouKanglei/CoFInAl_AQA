from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.transformer import Transformer
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def eth( in_channels, num_classes, normal = False):
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
    if normal:
        etf_vec = (etf_vec / torch.sum(etf_vec, dim=0, keepdim=True))
    return etf_vec, etf_rect


class GDLT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions = None):
        super(GDLT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        # torch.manual_seed(3407)
        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).to(device)
    def forward(self, x, return_feat = False):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)

        s1 = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s1, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        if return_feat:
            return {'output': out, 'embed': q1, 'other':{"s":s1.clone().cpu().detach().numpy()}}
        return {'output': out, 'embed': q1, 's':s1}

def choose_activate(type_id, n_query, device):
    if type_id == 0:
        return torch.linspace(0, 1, n_query, requires_grad=False).to(device), torch.linspace(0, 1, n_query, requires_grad=False).to(device).flip(-1)
    elif type_id == 1:
        return torch.linspace(0, 1, n_query+1, requires_grad=False)[1:].to(device), torch.linspace(0, 1, n_query+1, requires_grad=False)[1:].to(device).flip(-1)
    elif type_id == 2:
        # sigmoid
        return torch.tensor([0.1, 0.2, 0.8, 1], requires_grad=False).to(device).to(torch.float32), torch.tensor([1, 1, -1, -1], requires_grad=False).to(device).to(torch.float32)
    else:
        # arcl1
        d = [-1, -0.8, 0.8, 1]
        return torch.tensor(d, requires_grad=False).to(device).to(torch.float32),torch.tensor( d, requires_grad=False).to(device).to(torch.float32).flip(-1)

class cofinal(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions):
        super(cofinal, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),    
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # old 
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.regressor_revert = nn.Linear(hidden_dim, n_query)
        we = choose_activate(activate_regular_restrictions, n_query, device)
        self.weight = we[0] 
        self.weight_revert = we[1] 

        # etf
        self.val = False
        self.score_len = 100        
        etf_vec1, etf_rect = eth( self.score_len, self.score_len) 
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect
 

        self.key_len = 1
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for i in range(self.key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, 256),
                    nn.Linear(hidden_dim, self.score_len),
                )
            )
            self.regi.append(
                torch.nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 4, dropout = dropout),
            )

    def forward(self, x, return_feat = False):
        # x (b, t, c)
        result = {
            "int": None,
            "int_revert": None,
            "dec": [],
        }
        
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)   
        q1 = self.transformer.decoder(q, encode_x) # torch.Size([32, 4, 256])

        s = self.regressor(q1) 
        out = self.pre_old(s, self.weight, b)
        result['int'] = out

        s1 = self.regressor_revert(q1) 
        out = self.pre_old(s1, self.weight_revert, b)
        result['int_revert'] = out
        
        # q1 = q1.view(b, -1)
        # s = self.regressor1(q1) # - s.clone().cpu().detach() ( 32, 4 * 256)
        for i in range(self.key_len):
            s_dec, _ = self.regi[i](q1,q1,q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.pre_logits(s_dec)
            result['dec'].append(norm_d)

        if return_feat:
            return {'output': result, 'embed': q1, "other":{
                "int":s.clone().cpu().detach().numpy() ,
                "int_revert":s1.clone().cpu().detach().numpy(),
                "dec":s_dec.clone().cpu().detach().numpy()}
            }

        return {'output': result, 'embed': q1}
    
    def pre_old(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 10000).long()
        
        g_1 = gt_label//100 / 100
        g_2 = gt_label - gt_label//100 * 100
        # print(gt_label1, g_1, g_2)
        target_1 = self.etf_vec1[:, g_2].t()
        target = [ g_1, torch.ones_like(g_1) - g_1, target_1]
        # print(g_1, torch.ones_like(g_1) - g_1, g_2)
        return target

    def get_score(self, x):
        # print(x)
        cls_score1 = x['dec'][0] @ self.etf_vec1 # @ x[0].T
        c_1 = self.re_proj(cls_score1)
        score = (x['int'] + torch.ones_like(x['int_revert']) - x['int_revert'])/2  +  c_1/10000
        # print(x['int'] ,torch.ones_like(x['int_revert']) - x['int_revert'] , c_1, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target

class GDLTETH2(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, activate_regular_restrictions):
        super(GDLTETH2, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),    
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        # old 
        self.eval_classes = n_query
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        we = choose_activate(activate_regular_restrictions, n_query, device)
        self.weight = we[0] 

        # etf
        self.val = False
        self.score_len = 100        
        etf_vec1, etf_rect = eth( self.score_len, self.score_len) # 1000 score etf_rect[:,4]
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect
 
        self.key_len = 1
        self.regressori = nn.ModuleList()
        self.regi = nn.ModuleList()
        for i in range(self.key_len):
            self.regressori.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4 * hidden_dim, 256),
                    nn.Linear(hidden_dim, self.score_len),
                )
            )
            self.regi.append(
                torch.nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 4, dropout = dropout),
            )

    def forward(self, x, return_feat = False):
        # x (b, t, c)
        result = {
            "int": None,
            "dec": [],
        }

        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)   
        q1 = self.transformer.decoder(q, encode_x) # torch.Size([32, 4, 256])

        s = self.regressor(q1) 
        out = self.pre_old(s, self.weight, b)
        result['int'] = out
        
        # q1 = q1.view(b, -1)
        # s = self.regressor1(q1) # - s.clone().cpu().detach() ( 32, 4 * 256)
        for i in range(self.key_len):
            s_dec,_ = self.regi[i](q1,q1,q1)
            s_dec = self.regressori[i](s_dec)
            norm_d = self.pre_logits(s_dec)
            result['dec'].append(norm_d)
        if return_feat:
            return {'output': result, 'embed': q1, "other":{ "int":s.clone().cpu().detach().numpy() , "dec":s_dec.clone().cpu().detach().numpy()}}
        return {'output': result, 'embed': q1}
    
    def pre_old(self, s, weight, b):
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) torch.Size([32, 4, 4])
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return out
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x / torch.sum(x, dim=1, keepdim=True)
        return x
    
    def re_proj(self, x):
        return torch.argmax(x)

    def re_proj1(self, x, gt_label = None):
        x = torch.bmm( x.unsqueeze(1) , self.ref_poj[gt_label].unsqueeze(-1))
        return x

    def get_proj_class(self, gt_label1):
        gt_label = (gt_label1 * 10000).long()
        
        g_1 = gt_label//100 / 100
        g_2 = gt_label - gt_label//100 * 100
        # print(gt_label1, g_1, g_2)
        target_1 = self.etf_vec1[:, g_2].t()
        target = [ g_1, target_1]
        return target

    def get_score(self, x):
        # print(x)
        cls_score1 = x['dec'][0] @ self.etf_vec1 # @ x[0].T
        c_1 = self.re_proj(cls_score1)
        score = x['int']  +  c_1/10000
        # print(c_1, c_2, score)
        return score

    def eth_head(self, x = None, gt_label = None):
        target = 0
        if gt_label is None:
            target = self.get_score(x)
        if x is None:
            target = self.get_proj_class(gt_label)
        return target



