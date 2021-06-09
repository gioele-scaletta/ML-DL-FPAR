import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import numpy as np

class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()
        self.d_k = 64
        self.d_v = 64
        self.d_model = 512
        self.d_ff = 2048
        self.heads = 8
        self.W_q = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(torch.cuda.DoubleTensor), requires_grad=True))
        self.W_k = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_k).type(torch.cuda.DoubleTensor), requires_grad=True))
        self.W_v = torch.nn.init.xavier_normal_(Variable(torch.randn(self.d_model, self.d_v).type(torch.cuda.DoubleTensor), requires_grad=True))
        self.W_o = torch.nn.init.xavier_normal_(Variable(torch.randn(self.heads*self.d_v, self.d_model).type(torch.cuda.DoubleTensor), requires_grad=True))
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
        frame_position = self.positionalencoding(frame.size(0))
        frame2 = torch.add(frame, frame_position)
        #print(frame.size())
        multi_head_output = self.temporal_attention(frame.squeeze(0).type(torch.cuda.DoubleTensor))
        transformer_output = self.MLP_head(multi_head_output.cuda())
        return transformer_output


    def temporal_attention(self,frame):
        outputs = []
        for i in range(self.heads):
            Query = torch.matmul(frame,self.W_q)
            Key = torch.matmul(frame,self.W_k)
            Value = torch.matmul(frame,self.W_v)

            single_head_output = self.self_attention(Query,Key,Value)
            outputs.append(single_head_output)
        heads_concat_output = torch.cat(outputs,axis=1)

        multi_head_output = torch.matmul(heads_concat_output,self.W_o.cuda())
        multi_head_output = torch.add(multi_head_output, frame.cuda())
        layer_normalization1 = torch.nn.LayerNorm(512).cuda()
        multi_head_output = layer_normalization1(multi_head_output.type(torch.cuda.FloatTensor)) #the error says "RuntimeError: expected scalar type Double but found Float" but actually works if I cast to float from double, bello
            
        
        multi_head_output = torch.unsqueeze(multi_head_output,2)
        return multi_head_output


    def self_attention(self,Query,Key,Value):
        Key_T = torch.transpose(Key,0,1)
        Query_Key_T = torch.div(torch.matmul(Query,Key_T),math.sqrt(self.d_k))
        attention = F.softmax(Query_Key_T.type(torch.cuda.DoubleTensor),dim=-1)
        self_attention_output = torch.matmul(attention,Value.cuda())
        return self_attention_output


    def MLP_head(self,multi_head_output):
        out = self.classifier(multi_head_output.type(torch.cuda.FloatTensor))
        out = torch.add(out,multi_head_output)
        
        layer_normalization2 = torch.nn.LayerNorm(512).cuda()

        out = layer_normalization2(out.squeeze(2).type(torch.cuda.FloatTensor))
        out = torch.mean(out,0).view(-1) #not really sure if mean is the best thing, BERT and VTN use the CLS token
        return out
    
    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.double(self.d_model))
        return pos * angle_rates

    def positionalencoding(self,position):
        # NOTE the code is from https://www.tensorflow.org/tutorials/text/transformer, not ours
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],np.arange(self.d_model)[np.newaxis, :],self.d_model)
  
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = torch.tensor(pos_encoding).cuda()
        return pos_encoding

