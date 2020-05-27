import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit
from .utils import check_numpy
from warnings import warn


class ODST(ModuleWithInit):
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=sparsemax, bin_function=sparsemoid,
                 initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_,                   #这里的初始化时随机的
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(
            torch.zeros([in_features, num_trees, depth]), requires_grad=True
        )
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)                                                                     #6，1-32
            offsets = 2 ** torch.arange(self.depth)                                                                     #64， 0-63
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)                              #bin_codes depth* 2**depth
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)                                     #one hot 不会改变（one_hot的作用是特征二值化）
            # ^-- [depth, 2 ** depth, 2]
                                                                                                                        # 6* 64 0101 0011 到 0....01....1 相当于形成决策树
    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]                                                                            输入的features对应的权重的矩阵
                                                #batch_size, in_features
        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)                                          #一种矩阵的运算        input是batch过后的输入
        # ^--[batch_size, num_trees, depth]             num_trees 可看成每个标签， depth看成标签每层的值

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)   #[num_trees, depth]

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)                                   #正负数在做了entmax过后加起来得到的值为1
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)                                                                      #这一步相当于是sparesmax调整权重
        # ^--[batch_size, num_trees, depth, 2], approximately binary
            #[depth, 2 ** depth, 2]    2 被求和×了（赋值权重）
        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)                                         #和下面的一起看（单个路径的概率）
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)                                                              #相当于depth维度被压缩成一个值（通过乘积的形式） 即 通过底层（）2**depth 得到权重
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]             response :zeros [num_trees, tree_dim, 2 ** depth]
        # 2**depth 求和，相当于
        return response.flatten(1, 2) if self.flatten_output else response                                              #相当于是否将输出全连接化


    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():                                                                                           #不要grad
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,                    #Beta distribution
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),       #np.percentile是求n%的值
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)                                                                          #(没有带上batch_size所以无所谓)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)                            #模拟退火算法

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )

