import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

#manca positional embedding

def transformer(frame):
    multi_head_output = temporal_attention(frame)
    transformer_output = MLP_head(multi_head_output)
    return transformer_output


def temporal_attention(frame):
    d_k = 16 #num_frames
    d_v = 16 #num_frames
    d_model = 512 #size of the embedding, equal to the number of channels of the final resnet stage (after conv_5)
    heads = 8 #number of heads to be optimized
    outputs = []
    for i in range(heads):
        W_q = torch.nn.init.xavier_normal_(Variable(torch.randn(d_model, d_k).type(dtype=torch.float32), requires_grad=True))
        W_k = torch.nn.init.xavier_normal_(Variable(torch.randn(d_model, d_k).type(dtype=torch.float32), requires_grad=True))
        W_v = torch.nn.init.xavier_normal_(Variable(torch.randn(d_model, d_v).type(dtype=torch.float32), requires_grad=True))
        
        Query = torch.matmul(query,W_q)
        Key = torch.matmul(key,W_k)
        Value = torch.matmul(value,W_v)
        
        single_head_output = self_attention(Query,Key,Value)
        outputs.append(single_head_output)
    heads_concat_output = torch.cat(outputs,axis=1)
    
    W_o = torch.nn.init.xavier_normal_(Variable(torch.randn(heads*d_v, d_model).type(dtype=torch.float32), requires_grad=True))
    multi_head_output = torch.matmul(heads_concat_output,W_o)
    multi_head_output = torch.unsqueeze(multi_head_output,2)
    return multi_head_output
 

def self_attention(Query,Key,Value):
    d_k = 16 #num_frames
    Key_T = torch.transpose(Key)
    Query_Key_T = torch.div(torch.matmul(Query,Key_T),torch.sqrt(torch.cast(d_k,dtype=torch.float32)))
    attention = torch.nn.softmax(Query_Key_T,axis=-1)
    self_attention_output = torch.matmul(attention,Value)
    return self_attention_output

  
def MLP_head(multi_head_output):
    d_ff = 2048 #this is taken from the paper "Attention is All You Need" Vaswani et al.
    d_model = 512 #size of the embedding, equal to the number of channels of the final resnet stage (after conv_5)
    n_classes = 61
    out = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.gelu(),
      nn.Dropout(0.1),
      nn.Linear(d_ff, d_model)
    )
    out = torch.add(out,multi_head_output)
    return out
