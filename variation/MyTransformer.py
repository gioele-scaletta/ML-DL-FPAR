import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math

#manca positional embedding
#layer normalization

class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()
        self.d_k = 64
        self.d_v = 64
        self.d_model = 512
        self.d_ff = 2048
        self.heads = 8
        self.W_q = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(dtype=torch.float32), requires_grad=True))
        self.W_k = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(dtype=torch.float32), requires_grad=True))
        self.W_v = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_v).type(dtype=torch.float32), requires_grad=True))
        self.W_o = torch.nn.init.xavier_normal_(Variable(torch.randn(self.heads*self.d_v, self.d_model).type(dtype=torch.float32), requires_grad=True))
        self.conv1 = nn.Conv1d(self.d_model, self.d_ff, kernel_size=1, stride=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(self.d_ff, self.d_model, kernel_size=1, stride=1)
        self.classifier = nn.Sequential(self.conv1,self.activation,self.dropout,self.conv2)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)



    def forward(self,frame):
        frame = frame + positional_encoding(7,512) 
        multi_head_output = self.temporal_attention(frame)
        transformer_output = self.MLP_head(multi_head_output)
        return transformer_output


    def temporal_attention(self,frame):
        outputs = []
        for i in range(self.heads):
            Query = torch.matmul(frame,self.W_q)
            Key = torch.matmul(frame,self.W_k)
            Value = torch.matmul(frame,self.W_v)

            single_head_output = self.self_attention(Query,Key,Value)
            single_head_output = torch.add(single_head_output, frame)
            single_head_output = torch.nn.LayerNorm(single_head_output)
            outputs.append(single_head_output)
        heads_concat_output = torch.cat(outputs,axis=1)

        multi_head_output = torch.matmul(heads_concat_output,self.W_o.cuda())
        multi_head_output = torch.unsqueeze(multi_head_output,2)
        return multi_head_output


    def self_attention(self,Query,Key,Value):
        Key_T = torch.transpose(Key)
        Query_Key_T = torch.div(torch.matmul(Query,Key_T),math.sqrt(self.d_k))
        attention = F.softmax(Query_Key_T,dim=-1)
        self_attention_output = torch.matmul(attention,Value)
        return self_attention_output


    def MLP_head(self,multi_head_output):
        out = self.classifier(multi_head_output)
        out = torch.add(out,multi_head_output)
        out = torch.nn.LayerNorm(out)
        out = torch.mean(out,1).view(self.d_model) #not really sure if mean is the best thing, BERT and VTN use the CLS token
        return out
    
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positionalencoding(position, d_model):
        # NOTE the code is from https://www.tensorflow.org/tutorials/text/transformer, not ours
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
        pos_encoding = angle_rads[np.newaxis, ...]
    
        return pos_encoding
