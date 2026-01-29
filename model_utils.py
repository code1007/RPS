from os.path import join
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from typing import Optional, Tuple


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


from torch.nn.functional import *

def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads,
                                in_proj_weight, in_proj_bias, bias_k, bias_v,
                                add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
                                training=True, key_padding_mask=None, need_weights=True,
                                attn_mask=None, use_separate_proj_weight=False,
                                q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
                                static_k=None, static_v=None, need_raw=False, chunk_size=None):
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            if query.dtype != in_proj_weight.dtype:
                query = query.to(in_proj_weight.dtype)
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif torch.equal(key, value):
            # encoder-decoder attention
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            if query.dtype != _w.dtype:
                query = query.to(_w.dtype)
            q = linear(query, _w, _b)
            
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                if key.dtype != _w.dtype:
                    key = key.to(_w.dtype)
                k, v = linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            if query.dtype != _w.dtype:
                query = query.to(_w.dtype)
            q = linear(query, _w, _b)
            
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            if key.dtype != _w.dtype:
                key = key.to(_w.dtype)
            k = linear(key, _w, _b)
            
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            if value.dtype != _w.dtype:
                value = value.to(_w.dtype)
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        
        if query.dtype != q_proj_weight_non_opt.dtype:
            query = query.to(q_proj_weight_non_opt.dtype)
        if key.dtype != k_proj_weight_non_opt.dtype:
            key = key.to(k_proj_weight_non_opt.dtype)
        if value.dtype != v_proj_weight_non_opt.dtype:
            value = value.to(v_proj_weight_non_opt.dtype)
        
        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, None)
            k = linear(key, k_proj_weight_non_opt, None)
            v = linear(value, v_proj_weight_non_opt, None)
    
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    if chunk_size is not None and tgt_len > chunk_size:
        q_chunks = torch.split(q, chunk_size, dim=0)
        attn_output_chunks = []
        
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        src_len = k.size(1)
        
        for i, q_chunk in enumerate(q_chunks):
            chunk_len = q_chunk.size(0)
            
            q_chunk = q_chunk.contiguous().view(chunk_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            attn_weights_chunk = torch.bmm(q_chunk, k.transpose(1, 2))
            
            if attn_mask is not None:
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_len
                if attn_mask.dim() == 2:
                    mask_chunk = attn_mask[start_idx:end_idx, :]
                    attn_weights_chunk += mask_chunk.unsqueeze(0)
                else:
                    mask_chunk = attn_mask[:, start_idx:end_idx, :]
                    attn_weights_chunk += mask_chunk
            
            if key_padding_mask is not None:
                attn_weights_chunk = attn_weights_chunk.view(bsz, num_heads, chunk_len, src_len)
                attn_weights_chunk = attn_weights_chunk.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_weights_chunk = attn_weights_chunk.view(bsz * num_heads, chunk_len, src_len)
            
            attn_weights_chunk = softmax(attn_weights_chunk, dim=-1)
            attn_weights_chunk = dropout(attn_weights_chunk, p=dropout_p, training=training)
            
            attn_output_chunk = torch.bmm(attn_weights_chunk, v)
            attn_output_chunk = attn_output_chunk.transpose(0, 1).contiguous().view(chunk_len, bsz, embed_dim)
            attn_output_chunk = linear(attn_output_chunk, out_proj_weight, out_proj_bias)
            
            attn_output_chunks.append(attn_output_chunk)
            
            del q_chunk, attn_weights_chunk, attn_output_chunk
            torch.cuda.empty_cache()
        
        attn_output = torch.cat(attn_output_chunks, dim=0)
        
        if need_weights:
            return attn_output, None
        else:
            return attn_output, None
    
    if attn_mask is not None:
        assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:

            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output,attn_output_weights_raw

            #attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            #return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        result = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
            attn_mask=attn_mask, chunk_size=1024) # Pass chunk_size here
        if isinstance(result, tuple):
            attn_output, attn_weights = result
        else:
            attn_output, attn_weights = result, None

        return attn_output, attn_weights

class MultiheadAttention(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, need_raw=False, chunk_size=None):
        
        if chunk_size is None:
            chunk_size = 64  
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, chunk_size=chunk_size)
        else:
            attn_output, attn_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, chunk_size=chunk_size)

        return attn_output, attn_weights


# for graph construction
import nmslib
import math
class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(coords, features, threshold=5000, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(coords.cpu().detach()), np.array(features.cpu().detach())
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    start_point = edge_spatial[0, :]
    end_point = edge_spatial[1, :]
    start_coord = coords[start_point]
    end_coord = coords[end_point]
    tmp = start_coord - end_coord
    edge_distance = []
    for i in range(tmp.shape[0]):
        distance = math.hypot(tmp[i][0], tmp[i][1])
        edge_distance.append(distance)

    filter_edge_spatial = edge_spatial[:, np.array(edge_distance) <= threshold]

    G = geomData(x = torch.Tensor(features),
                 edge_index = filter_edge_spatial,
                 edge_latent = edge_latent,
                 centroid = torch.Tensor(coords))

    return G


def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy
