import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

#manca positional embedding
#layer normalization

class MyConvLSTMCell(nn.Module):
    def __init__(self):
        super(MyConvLSTMCell, self).__init__()
        self.d_k = 16
        self.d_v = 16
        self.d_model = 512
        self.d_ff = 2048
        self.heads = 8
        self.W_q = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(dtype=torch.float32), requires_grad=True))
        self.W_k = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(dtype=torch.float32), requires_grad=True))
        self.W_v = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_v).type(dtype=torch.float32), requires_grad=True))
        self.W_o = torch.nn.init.xavier_normal_(Variable(torch.randn(heads*self.d_v, self.d_model).type(dtype=torch.float32), requires_grad=True))
        self.fc1 = nn.Linear(self.d_model, self.d_ff)
        self.activation = nn.gelu()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.d_ff, self.d_model)
        self.classifier = nn.Sequential(self.fc1,self.activation,self.dropout,self.fc2)


    def forward(self,frame):
        multi_head_output = temporal_attention(frame)
        transformer_output = MLP_head(multi_head_output)
        return transformer_output


    def temporal_attention(self,frame):
        outputs = []
        for i in range(heads):
            Query = torch.matmul(query,self.W_q)
            Key = torch.matmul(key,self.W_k)
            Value = torch.matmul(value,self.W_v)

            single_head_output = self_attention(Query,Key,Value)
            outputs.append(single_head_output)
        heads_concat_output = torch.cat(outputs,axis=1)

        multi_head_output = torch.matmul(heads_concat_output,self.W_o)
        multi_head_output = torch.unsqueeze(multi_head_output,2)
        return multi_head_output


    def self_attention(self,Query,Key,Value):
        Key_T = torch.transpose(Key)
        Query_Key_T = torch.div(torch.matmul(Query,Key_T),torch.sqrt(torch.cast(self.d_k,dtype=torch.float32)))
        attention = torch.nn.softmax(Query_Key_T,axis=-1)
        self_attention_output = torch.matmul(attention,Value)
        return self_attention_output


    def MLP_head(self,multi_head_output):
        out = self.classifier(multi_head_output)
        out = torch.add(out,multi_head_output)
        return out
