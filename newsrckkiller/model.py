import torch
import torch.nn as nn
import timm
import numpy as np
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHead(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.w_qs.bias.data.fill_(0)
        self.w_ks.bias.data.fill_(0)
        self.w_vs.bias.data.fill_(0)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()   # (batch_size, 80, 512)
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (batch_size, T, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk, (batch_size*8, T, 64)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)   # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)
        
        output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv), (batch_size, T, 512)
        output = F.relu_(self.dropout(self.fc(output)))
        return output
class Time2Vec(nn.Module):
   """ Time2Vec
   Inspired of : https://github.com/ojus1/Time2Vec-PyTorch
   and https://discuss.pytorch.org/t/how-to-get-the-batch-dimension-right-in-the-forward-path-of-a-custom-layer/80131/2
   Original paper : https://arxiv.org/pdf/1907.05321.pdf
   Keras implementation : https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
   """

   def __init__(self, input_dim, output_dim):
      super().__init__()
      self.output_dim = output_dim
      self.input_dim = input_dim
      self.w0 = nn.Parameter(torch.Tensor(1, input_dim))
      self.phi0 = nn.Parameter(torch.Tensor(1, input_dim))
      self.W = nn.Parameter(torch.Tensor(input_dim, output_dim-1))
      self.Phi = nn.Parameter(torch.Tensor(input_dim, output_dim-1))
      self.reset_parameters()

   def reset_parameters(self):
      nn.init.uniform_(self.w0, 0, 1)
      nn.init.uniform_(self.phi0, 0, 1)
      nn.init.uniform_(self.W, 0, 1)
      nn.init.uniform_(self.Phi, 0, 1)

   def forward(self, x):
      n_batch = x.size(0)
      original = (x*self.w0 + self.phi0).unsqueeze(-1)
      x = torch.repeat_interleave(x, repeats=self.output_dim-1, dim=0).view(n_batch,-1,self.output_dim-1)
      x = torch.sin(x * self.W + self.Phi)
      return torch.cat([original,x],-1).view(n_batch,self.output_dim,-1).contiguous()


class resnest_meta(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.base = timm.create_model("resnest50d_1s4x24d", pretrained=True, num_classes=0)
        
        ### META DATA###
        self.num_meta = 5
        self.meta_emb = 15 #64
        self.n_head = 8
        self.d_k = self.d_v =128
        self.dropout_transfo = 0.3
        self.t2v = Time2Vec(self.num_meta, self.meta_emb)
        self.multihead_meta = MultiHead(self.n_head, self.num_meta, self.d_k, self.d_v, self.dropout_transfo)
        self.fc = nn.Linear(2048 + 75, num_classes)

    def forward(self, image, meta):
        image = self.base(image)
        meta = self.t2v(meta)
        meta = self.multihead_meta(meta, meta, meta)
        meta = meta.view((-1, meta.size(1) * meta.size(2)))
        image = torch.cat([image, meta], axis=1)
        image = self.fc(image)
        return   image

class eff_meta(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.base = timm.create_model("tf_efficientnet_b2_ns", pretrained=True, num_classes=0)
        
        ### META DATA###
        self.num_meta = 5
        self.meta_emb = 15 #64
        self.n_head = 8
        self.d_k = self.d_v =128
        self.dropout_transfo = 0.3
        self.t2v = Time2Vec(self.num_meta, self.meta_emb)
        self.multihead_meta = MultiHead(self.n_head, self.num_meta, self.d_k, self.d_v, self.dropout_transfo)
        self.fc = nn.Linear(1408 + 75, num_classes)#b2 - 1408

    def forward(self, image, meta):
        image = self.base(image)
        meta = self.t2v(meta)
        meta = self.multihead_meta(meta, meta, meta)
        meta = meta.view((-1, meta.size(1) * meta.size(2)))
        image = torch.cat([image, meta], axis=1)
        image = self.fc(image)
        return   image

class dense_meta(nn.Module):
    def __init__(self, num_classes=24, pretrained=True):
        super().__init__()
        self.base = timm.create_model("densenet201", pretrained=pretrained, num_classes=0)
        
        ### META DATA###
        self.num_meta = 5
        self.meta_emb = 15 #64
        self.n_head = 8
        self.d_k = self.d_v =128
        self.dropout_transfo = 0.3
        self.t2v = Time2Vec(self.num_meta, self.meta_emb)
        self.multihead_meta = MultiHead(self.n_head, self.num_meta, self.d_k, self.d_v, self.dropout_transfo)
        self.fc = nn.Linear(1920 + 75, num_classes)#b2 - 1408

    def forward(self, image, meta):
        image = self.base(image)
        meta = self.t2v(meta)
        meta = self.multihead_meta(meta, meta, meta)
        meta = meta.view((-1, meta.size(1) * meta.size(2)))
        image = torch.cat([image, meta], axis=1)
        image = self.fc(image)
        return   image

def get_model(name, num_classes=397):
    """
    Loads a pretrained model. 
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if name.startswith("resnest"):
        model = resnest_meta(num_classes)
    elif name.startswith("tf_efficientnet_b"):
        model = eff_meta(num_classes)
    elif name.startswith("densenet"):
        model = dense_meta(num_classes)
    else:
        pass
    return model
