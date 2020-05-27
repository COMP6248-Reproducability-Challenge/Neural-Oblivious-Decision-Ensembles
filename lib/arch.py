import torch
import torch.nn as nn
import torch.nn.functional as F
from .odst import ODST


class DenseBlock(nn.Sequential):        #layer_dim == num_trees
    def __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None,                                 #可提升的地方：max——feature 以及 dropout的参数
                 input_dropout=0.0, flatten_output=True, Module=ODST, **kwargs):
        layers = []
        for i in range(num_layers):     #layer的次数在此
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs)
            # response :[batch_size, num_trees, tree_dim]
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))                             #max_feature 为dense net进来的上限 结合上面init，进来的时候为无穷大
            layers.append(oddt)                     #这个时候用了tree_dim

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        #print(initial_features)  torch.Size([2000, 11])   11
        for layer in self:      #即对layer次数做forward，上面有layer次数的存档
            layer_inp = x
            if self.max_features is not None:                                                                           #如果有最大feature的限制的话，这一步会使feature限制在一个给定的值
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)                                                    #没有dropout掉（因为赋值为0.0）
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)          #cat的作用只是拼接                                                    #结合上面，即为添加dropout函数

        outputs = x[..., initial_features:]                                                                             #省略号包括前面所有维度的数据 输出的东西接在之前的features之后
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)                #[2000,200,3] tree dim 为num_class + 1
        #print(outputs.size())                              2                   1024                                    flatten 的作用是将多维输出一维话，用作全连接
        return outputs



