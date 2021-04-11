
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy

@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)


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

class Normed_Linear(nn.Linear):
    """ Linear Layer with weight and input L2 normalized
    Could lead to better 'geometric' space and could deal with imbalance dataset issues
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        bias (bool) : If False, the layer will not learn an additive bias.
    Shape:
        Input: (N, *, in_features)
        Output: (N, *, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        weight = self.weight/(torch.norm(((self.weight)), 2, 0) + 1e-5)
        x = x/(torch.norm(((x)), 2, -1)+1e-5).unsqueeze(-1)
        return F.linear(x, weight, self.bias)

class AvgMaxPool2d(nn.Module):
    """ Average + Max Pooling layer
    Average Pooling added to Max Pooling
    Args:
        pool_stride (int, tuple) : controls the pooling stride
    """
    def __init__(self, pool_stride):
        super().__init__()
        self.pool_stride = pool_stride
        self.avgpool = nn.MaxPool2d(self.pool_stride)
        self.maxpool = nn.AvgPool2d(self.pool_stride)
    
    def forward(self, x):
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        return x1+x2

class Pooling_Head(nn.Module):
    """ Pooling layer for MIL
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        pooling (str) : pooling strategie, can be max, ave, lin, exp, att, auto
    """
    def __init__(self, in_features, out_features, pooling):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooling = pooling

        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.in_features, self.out_features)
            nn.init.xavier_uniform_(self.fc_att.weight); nn.init.constant_(self.fc_att.bias, 0)
        elif self.pooling == 'auto':
            self.autopool = AutoPool(self.out_features)
    
    def forward(self, frame_prob, x):
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob #, frame_att
        elif self.pooling == 'auto':
            global_prob = self.autopool(frame_prob)
            return global_prob, frame_prob


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
#%%


class AutoPool(nn.Module):
   """ Adaptive pooling operators for Multiple Instance Learning
   Adapted original code.
   This layer automatically adapts the pooling behavior to interpolate
   between min, mean and max-pooling for each class.
   Link : https://github.com/marl/autopool
   Args:
      input_size (int): Lenght of input_vector
      time_axis (int): Axis along which to perform the pooling. 
         By default 1 (should be time) ie. (batch_size, time_sample_size, input_size)
   """
   
   def __init__(self, input_size, time_axis=1):
      super(AutoPool, self).__init__()
      self.time_axis = time_axis
      self.alpha = nn.Parameter(torch.zeros(1, input_size))

   def forward(self, x):
      scaled = self.alpha*x
      weights = F.softmax(scaled, dim=self.time_axis)
      return (x * weights).sum(dim=self.time_axis)

class TimeDistributed(nn.Module):
   """ Takes an operation and applies it for each time sample
   Ref: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
   Args:
      module (nn.Module): The operation
      batch_first (bool): If true, x is (samples, timesteps, output_size)
         Else, x is (timesteps, samples, output_size)
   """

   def __init__(self, module, batch_first=True):
      super(TimeDistributed, self).__init__()
      self.module = module
      self.batch_first = batch_first

   def forward(self, x):

      if len(x.size()) <= 2:
         return self.module(x)
      # Squash samples and timesteps into a single axis
      x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
      y = self.module(x_reshape)
      # We have to reshape Y
      if self.batch_first:
         y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
      else:
         y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
      return y

class DCASE_Baseline(nn.Module):
   """ DCASE Baseline model
   Note: 
      - inputs size is 512 embeddings generated by openl3, 2 spatial metadata 
      and 83 time metadata (3 hot encoded metadata)
      - L2 regularization of 1e-5 (except on AutoPool)
      - Adam optimizer
      - Trained on 100 epoch max
      - Early stopping on validation loss (patience = 20)
   Args:
      input_size (int): size of the input vector
      num_classes (int): number of classes
      hidden_layer_size (int): size of hidden layer
      num_hidden_layers (int): number of hidden layers
   """

   def __init__(self, input_size, num_classes, hidden_layer_size=128, num_hidden_layers=1):
      super(DCASE_Baseline, self).__init__()
      self.layer1 = TimeDistributed(nn.Linear(input_size,hidden_layer_size))
      self.activation1 = TimeDistributed(nn.ReLU())
      self.layer2 = TimeDistributed(nn.Linear(hidden_layer_size,num_classes))
      self.activation2 = TimeDistributed(nn.Sigmoid())
      self.autopool = AutoPool(num_classes)
   
   def forward(self, x):
      x = self.layer1(x)
      x = self.activation1(x)
      x = self.layer2(x)
      x = self.activation2(x)
      x = self.autopool(x)
      return x