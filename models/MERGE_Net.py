import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.transformer import ViT
from einops import rearrange, repeat
import spconv.pytorch as spconv


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=20):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        m_batchsize, C, class_num = x.size()

        proj_query = x
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        dynamic_adj = torch.bmm(proj_key, proj_query)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)

        return dynamic_adj, out

    def forward_dynamic_gcn(self, x, dynamic_adj):
        weight = self.dynamic_weight(x)

        support = torch.mul(x, weight)
        x = torch.matmul(support, dynamic_adj)
        x = self.relu(x)
        return x

    def forward(self, x, sds=[0, 1, 0]):
        static, dynamic, static_dynamic = sds
        if static:
            out_static = self.forward_static_gcn(x)

        if dynamic:
            dynamic_adj, out = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x, dynamic_adj)

        if static_dynamic:
            out_static = self.forward_static_gcn(x)
            x = x + out_static
            dynamic_adj, out = self.forward_construct_dynamic_graph(x)
            x = self.forward_dynamic_gcn(x, dynamic_adj)

        return x, out


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)

            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value,
                                       stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class CircleConv2d(nn.Module):
    def __init__(self, nin, nout, k=3, stride=1):
        super().__init__()

        self.register_parameter("weight", nn.Parameter(torch.randn([nout, nin, k, k]), requires_grad=True))
        self.register_parameter("bias", nn.Parameter(torch.zeros([nout]),requires_grad=True))

        self.register_buffer("mask", torch.ones([1, 1, k, k]))

        self.mask[0, 0, 0, 0] = 0
        self.mask[0, 0, 0, k-1] = 0
        self.mask[0, 0, k-1, 0] = 0
        self.mask[0, 0,k-1, k-1] = 0
        self.stride = stride
        self.pad = (k-1)//2
    def forward(self, x):
        w = self.weight * self.mask
        x = F.conv2d(x, weight=w, bias=self.bias, stride=self.stride, padding=self.pad)
        return x


class AnomalousConv3d(nn.Module):
    def __init__(self, nin, nout, c=3, k=3, stride=0):
        super().__init__()

        self.register_parameter("weight", nn.Parameter(torch.randn([nout, nin, c, k, k]), requires_grad=True))
        self.register_parameter("bias", nn.Parameter(torch.zeros([nout]),requires_grad=True))

        self.register_buffer("mask", torch.ones([1, 1, c, k, k]))

        self.mask[0, 0, 0, 0, 0] = 0
        self.mask[0, 0, 0, 0, k-1] = 0
        self.mask[0, 0, 0, k-1, 0] = 0
        self.mask[0, 0, 0, k-1, k-1] = 0
        self.mask[0, 0, -1, 0, 0] = 0
        self.mask[0, 0, -1, 0, k-1] = 0
        self.mask[0, 0, -1, k-1, 0] = 0
        self.mask[0, 0, -1, k-1, k-1] = 0
        self.stride = stride
        self.pad = (k-1)//2
    def forward(self, x):
        w = self.weight * self.mask

        x = F.conv3d(x, weight=w, bias=self.bias)

        return x


class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.ori_M = M

        self.LWS = LWS
        self.reg_ETF = reg_ETF

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class MERGE_Net(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes, patch_size, drop_prob=0.1, block_size=3, dataset_name='Berlin'):
        super(MERGE_Net, self).__init__()
        self.dataset_name = dataset_name
        self.input_channels = input_channels
        self.num_node = num_nodes
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                                         start_value=0.,
                                         stop_value=drop_prob,
                                         nr_steps=5e3)

        self.register_parameter("weight_conv1_3D", nn.Parameter(torch.randn([8, 1, 7, 3, 3]), requires_grad=True))
        self.register_parameter("bias_conv1_3D", nn.Parameter(torch.zeros([8]),requires_grad=True))

        self.register_parameter("weight_conv2_3D", nn.Parameter(torch.randn([16, 8, 5, 3, 3]), requires_grad=True))
        self.register_parameter("bias_conv2_3D", nn.Parameter(torch.zeros([16]),requires_grad=True))
        self.register_parameter("weight_conv3_3D", nn.Parameter(torch.randn([32, 16, 3, 3, 3]), requires_grad=True))
        self.register_parameter("bias_conv3_3D", nn.Parameter(torch.zeros([32]),requires_grad=True))
        self.register_buffer("mask_conv1_3D", torch.ones([1, 1, 1, 3, 3]))
        self.register_buffer("mask_conv2_3D", torch.ones([1, 1, 1, 3, 3]))
        self.register_buffer("mask_conv3_3D", torch.ones([1, 1, 3, 3, 3]))
        k = 3
        self.mask_conv1_3D[0, 0, 0, 0, 0] = 0
        self.mask_conv1_3D[0, 0, 0, 0, k-1] = 0
        self.mask_conv1_3D[0, 0, 0, k-1, 0] = 0
        self.mask_conv1_3D[0, 0, 0, k-1, k-1] = 0
        self.mask_conv1_3D[0, 0, -1, 0, 0] = 0
        self.mask_conv1_3D[0, 0, -1, 0, k-1] = 0
        self.mask_conv1_3D[0, 0, -1, k-1, 0] = 0
        self.mask_conv1_3D[0, 0, -1, k-1, k-1] = 0
        self.mask_conv2_3D[0, 0, 0, 0, 0] = 0
        self.mask_conv2_3D[0, 0, 0, 0, k-1] = 0
        self.mask_conv2_3D[0, 0, 0, k-1, 0] = 0
        self.mask_conv2_3D[0, 0, 0, k-1, k-1] = 0
        self.mask_conv2_3D[0, 0, -1, 0, 0] = 0
        self.mask_conv2_3D[0, 0, -1, 0, k-1] = 0
        self.mask_conv2_3D[0, 0, -1, k-1, 0] = 0
        self.mask_conv2_3D[0, 0, -1, k-1, k-1] = 0
        self.mask_conv3_3D[0, 0, 0, 0, 0] = 0
        self.mask_conv3_3D[0, 0, 0, 0, k-1] = 0
        self.mask_conv3_3D[0, 0, 0, k-1, 0] = 0
        self.mask_conv3_3D[0, 0, 0, k-1, k-1] = 0
        self.mask_conv3_3D[0, 0, -1, 0, 0] = 0
        self.mask_conv3_3D[0, 0, -1, 0, k-1] = 0
        self.mask_conv3_3D[0, 0, -1, k-1, 0] = 0
        self.mask_conv3_3D[0, 0, -1, k-1, k-1] = 0

        self.conv4_2D = nn.Conv2d(32, 64, 3)

        self.fc_1 = nn.Linear(141568, 256)

        self.drop1 = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(256, 128)

        self.drop2 = nn.Dropout(0.1)
        self.fc_3 = nn.Linear(128, self.num_classes)

        self.register_parameter("weight_conv1_3D_LiDAR", nn.Parameter(torch.randn([8, 1, 1, 3, 3]), requires_grad=True))

        self.register_parameter("bias_conv1_3D_LiDAR", nn.Parameter(torch.zeros([8]),requires_grad=True))
        self.register_parameter("weight_conv2_3D_LiDAR", nn.Parameter(torch.randn([16, 8, 1, 3, 3]), requires_grad=True))

        self.register_parameter("bias_conv2_3D_LiDAR", nn.Parameter(torch.zeros([16]),requires_grad=True))
        self.register_parameter("weight_conv3_3D_LiDAR", nn.Parameter(torch.randn([32, 16, 1, 3, 3]), requires_grad=True))

        self.register_parameter("bias_conv3_3D_LiDAR", nn.Parameter(torch.zeros([32]),requires_grad=True))
        self.register_buffer("mask_conv1_3D_LiDAR", torch.ones([1, 1, 1, 3, 3]))

        self.register_buffer("mask_conv2_3D_LiDAR", torch.ones([1, 1, 1, 3, 3]))

        self.register_buffer("mask_conv3_3D_LiDAR", torch.ones([1, 1, 1, 3, 3]))

        k = 3
        self.mask_conv1_3D_LiDAR[0, 0, 0, 0, 0] = 0
        self.mask_conv1_3D_LiDAR[0, 0, 0, 0, k-1] = 0
        self.mask_conv1_3D_LiDAR[0, 0, 0, k-1, 0] = 0
        self.mask_conv1_3D_LiDAR[0, 0, 0, k-1, k-1] = 0
        self.mask_conv1_3D_LiDAR[0, 0, -1, 0, 0] = 0
        self.mask_conv1_3D_LiDAR[0, 0, -1, 0, k-1] = 0
        self.mask_conv1_3D_LiDAR[0, 0, -1, k-1, 0] = 0
        self.mask_conv1_3D_LiDAR[0, 0, -1, k-1, k-1] = 0
        self.mask_conv2_3D_LiDAR[0, 0, 0, 0, 0] = 0
        self.mask_conv2_3D_LiDAR[0, 0, 0, 0, k-1] = 0
        self.mask_conv2_3D_LiDAR[0, 0, 0, k-1, 0] = 0
        self.mask_conv2_3D_LiDAR[0, 0, 0, k-1, k-1] = 0
        self.mask_conv2_3D_LiDAR[0, 0, -1, 0, 0] = 0
        self.mask_conv2_3D_LiDAR[0, 0, -1, 0, k-1] = 0
        self.mask_conv2_3D_LiDAR[0, 0, -1, k-1, 0] = 0
        self.mask_conv2_3D_LiDAR[0, 0, -1, k-1, k-1] = 0
        self.mask_conv3_3D_LiDAR[0, 0, 0, 0, 0] = 0
        self.mask_conv3_3D_LiDAR[0, 0, 0, 0, k-1] = 0
        self.mask_conv3_3D_LiDAR[0, 0, 0, k-1, 0] = 0
        self.mask_conv3_3D_LiDAR[0, 0, 0, k-1, k-1] = 0
        self.mask_conv3_3D_LiDAR[0, 0, -1, 0, 0] = 0
        self.mask_conv3_3D_LiDAR[0, 0, -1, 0, k-1] = 0
        self.mask_conv3_3D_LiDAR[0, 0, -1, k-1, 0] = 0
        self.mask_conv3_3D_LiDAR[0, 0, -1, k-1, k-1] = 0

        self.conv4_2D_LiDAR = nn.Conv2d(32, 64, 3)
        self.conv5_2D_LiDAR = nn.Conv2d(32, 64, (1, 3))

        self.fc_1_LiDAR = nn.Linear(20224, 256)

        self.drop1_LiDAR = nn.Dropout(0.2)
        self.fc_2_LiDAR = nn.Linear(256, 128)

        self.drop2_LiDAR = nn.Dropout(0.1)
        self.fc_3_LiDAR = nn.Linear(128, self.num_classes)

        self.fc_1_mean = nn.Linear(1352, 128)

        self.fc_1_mean_add = nn.Linear(1352, 128)

        self.fc_2_mean = nn.Linear(1936, 128)

        self.fc_2_mean_add = nn.Linear(1936, 128)

        self.fc_3_mean = nn.Linear(2592, 128)

        self.fc_3_mean_add = nn.Linear(2592, 128)

        self.fc_3_HSI_LiDAR = nn.Linear(128, self.num_classes)
        self.fc_4_HSI_LiDAR = nn.Linear(128 * 3, 128)
        self.fc_5_HSI_LiDAR = nn.Linear(128 * 3, 128)
        self.fc_6_HSI_LiDAR = nn.Linear(128 * 4, 128)
        self.fc_7_HSI_LiDAR = nn.Linear(128 * 3, self.num_classes)

        dim = 2212
        dim_LiDAR = 316
        if dataset_name == 'Berlin':
            dim = 4676
            dim_LiDAR = 668

            self.fc_1 = nn.Linear(299264, 256)
            self.fc_1_LiDAR = nn.Linear(42752, 256)
            self.fc_1_mean = nn.Linear(2312, 128)
            self.fc_1_mean_add = nn.Linear(2312, 128)
            self.fc_2_mean = nn.Linear(3600, 128)
            self.fc_2_mean_add = nn.Linear(3600, 128)
            self.fc_3_mean = nn.Linear(5408, 128)
            self.fc_3_mean_add = nn.Linear(5408, 128)
        elif dataset_name == 'Houston_2013' or dataset_name == 'Trento' or dataset_name == 'MUUFL':
            self.fc_1_LiDAR = nn.Linear(5056, 256)

            dim_LiDAR = 79

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, 64 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.en_transformer = Transformer(dim=dim, depth=5, heads=4, dim_head=16, mlp_head=8, dropout=0.1, num_channel=64)
        self.dropout = nn.Dropout(0.1)

        self.fc_1_x_tr = nn.Linear(65*dim, 256)
        self.fc_2_x_tr = nn.Linear(256, 128)

        self.encoder_pos_embed_LiDAR = nn.Parameter(torch.randn(1, 64 + 1, dim_LiDAR))
        self.cls_token_LiDAR = nn.Parameter(torch.randn(1, 1, dim_LiDAR))
        self.en_transformer_LiDAR = Transformer(dim=dim_LiDAR, depth=5, heads=4, dim_head=16, mlp_head=8, dropout=0.1, num_channel=64)

        self.fc_1_x_tr_LiDAR = nn.Linear(65*dim_LiDAR, 256)
        self.fc_2_x_tr_LiDAR = nn.Linear(256, 128)

        self.ratio = nn.Parameter(torch.randn(1))
        self.ratio1 = nn.Parameter(torch.randn(1))
        self.ratio2 = nn.Parameter(torch.randn(1))
        self.ratio3 = nn.Parameter(torch.randn(1))

    def forward(self, x, x_LiDAR, w):
        w1 = self.weight_conv1_3D * self.mask_conv1_3D
        w2 = self.weight_conv2_3D * self.mask_conv2_3D
        w3 = self.weight_conv3_3D * self.mask_conv3_3D

        x1 = F.leaky_relu(F.conv3d(x, weight=w1, bias=self.bias_conv1_3D, padding=(3, 0, 0)))

        x2 = F.leaky_relu(F.conv3d(x1, weight=w2, bias=self.bias_conv2_3D, padding=(2, 0, 0)))

        x3 = F.leaky_relu(F.conv3d(x2, weight=w3, bias=self.bias_conv3_3D))

        x3 = torch.reshape(x3, (x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3] * x3.shape[4]))

        x4 = F.leaky_relu(self.conv4_2D(x3))

        x_tr = x4

        x4 = torch.flatten(x4, start_dim=1)

        x4 = self.fc_1(x4)

        x4 = self.fc_2(x4)

        output = self.fc_3(x4)

        x_tr = torch.reshape(x_tr, (x_tr.shape[0], x_tr.shape[1], -1))
        b, n, _ = x_tr.shape

        x_tr = x_tr + self.encoder_pos_embed[:, 1:, :]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x_tr = torch.cat((cls_tokens, x_tr), dim=1)

        x_tr += self.encoder_pos_embed[:, :1]
        x_tr = self.dropout(x_tr)

        x_tr = self.en_transformer(x_tr, mask=None)

        x_tr = torch.flatten(x_tr, start_dim=1)
        x_tr = self.fc_1_x_tr(x_tr)
        x_tr = self.fc_2_x_tr(x_tr)

        if len(x_LiDAR.shape) < 5:
            x_LiDAR = torch.unsqueeze(x_LiDAR, 1)

        else:
            x_LiDAR = x_LiDAR.permute(0, 1, 4, 2, 3)

        w1_LiDAR = self.weight_conv1_3D_LiDAR * self.mask_conv1_3D_LiDAR
        w2_LiDAR = self.weight_conv2_3D_LiDAR * self.mask_conv2_3D_LiDAR
        w3_LiDAR = self.weight_conv3_3D_LiDAR * self.mask_conv3_3D_LiDAR

        x1_LiDAR = F.leaky_relu(F.conv3d(x_LiDAR, weight=w1_LiDAR, bias=self.bias_conv1_3D_LiDAR))

        x2_LiDAR = F.leaky_relu(F.conv3d(x1_LiDAR, weight=w2_LiDAR, bias=self.bias_conv2_3D_LiDAR))

        x3_LiDAR = F.leaky_relu(F.conv3d(x2_LiDAR, weight=w3_LiDAR, bias=self.bias_conv3_3D_LiDAR))

        x3_LiDAR = torch.reshape(x3_LiDAR, (
        x3_LiDAR.shape[0], x3_LiDAR.shape[1], x3_LiDAR.shape[2], x3_LiDAR.shape[3] * x3_LiDAR.shape[4]))

        x1_mean = torch.mean(x1, dim=2)

        x1_LiDAR_mean = torch.mean(x1_LiDAR, dim=2)

        x1_HSI_LiDAR_subtract_mean = abs(x1_mean - x1_LiDAR_mean)
        x1_HSI_LiDAR_add_mean = abs(x1_mean + x1_LiDAR_mean)

        x1_HSI_LiDAR_subtract_mean = torch.flatten(x1_HSI_LiDAR_subtract_mean, start_dim=1)
        x1_HSI_LiDAR_subtract_mean = self.fc_1_mean(x1_HSI_LiDAR_subtract_mean)
        x1_HSI_LiDAR_add_mean = torch.flatten(x1_HSI_LiDAR_add_mean, start_dim=1)
        x1_HSI_LiDAR_add_mean = self.fc_1_mean_add(x1_HSI_LiDAR_add_mean)

        x2_mean = torch.mean(x2, dim=2)

        x2_LiDAR_mean = torch.mean(x2_LiDAR, dim=2)

        x2_HSI_LiDAR_subtract_mean = abs(x2_mean - x2_LiDAR_mean)
        x2_HSI_LiDAR_add_mean = abs(x2_mean + x2_LiDAR_mean)

        x2_HSI_LiDAR_subtract_mean = torch.flatten(x2_HSI_LiDAR_subtract_mean, start_dim=1)
        x2_HSI_LiDAR_subtract_mean = self.fc_2_mean(x2_HSI_LiDAR_subtract_mean)
        x2_HSI_LiDAR_add_mean = torch.flatten(x2_HSI_LiDAR_add_mean, start_dim=1)
        x2_HSI_LiDAR_add_mean = self.fc_2_mean_add(x2_HSI_LiDAR_add_mean)

        x3_mean = torch.mean(x3, dim=2)

        x3_LiDAR_mean = torch.mean(x3_LiDAR, dim=2)

        x3_HSI_LiDAR_subtract_mean = abs(x3_mean - x3_LiDAR_mean)
        x3_HSI_LiDAR_add_mean = abs(x3_mean + x3_LiDAR_mean)

        x3_HSI_LiDAR_subtract_mean = torch.flatten(x3_HSI_LiDAR_subtract_mean, start_dim=1)
        x3_HSI_LiDAR_subtract_mean = self.fc_3_mean(x3_HSI_LiDAR_subtract_mean)
        x3_HSI_LiDAR_add_mean = torch.flatten(x3_HSI_LiDAR_add_mean, start_dim=1)
        x3_HSI_LiDAR_add_mean = self.fc_3_mean_add(x3_HSI_LiDAR_add_mean)

        x4_LiDAR = F.leaky_relu(self.conv5_2D_LiDAR(x3_LiDAR))

        x_tr_LiDAR = x4_LiDAR
        x_tr_LiDAR = torch.reshape(x_tr_LiDAR, (x_tr_LiDAR.shape[0], x_tr_LiDAR.shape[1], -1))
        b, n, _ = x_tr_LiDAR.shape

        x_tr_LiDAR = x_tr_LiDAR + self.encoder_pos_embed_LiDAR[:, 1:, :]

        cls_tokens = repeat(self.cls_token_LiDAR, '() n d -> b n d', b=b)
        x_tr_LiDAR = torch.cat((cls_tokens, x_tr_LiDAR), dim=1)

        x_tr_LiDAR += self.encoder_pos_embed_LiDAR[:, :1]
        x_tr_LiDAR = self.dropout(x_tr_LiDAR)

        x_tr_LiDAR = self.en_transformer_LiDAR(x_tr_LiDAR, mask=None)

        x_tr_LiDAR = torch.flatten(x_tr_LiDAR, start_dim=1)
        x_tr_LiDAR = self.fc_1_x_tr_LiDAR(x_tr_LiDAR)
        x_tr_LiDAR = self.fc_2_x_tr_LiDAR(x_tr_LiDAR)

        x4_LiDAR = torch.flatten(x4_LiDAR, start_dim=1)

        x4_LiDAR = self.fc_1_LiDAR(x4_LiDAR)

        x4_LiDAR = self.fc_2_LiDAR(x4_LiDAR)

        output_add_mean = torch.cat((x1_HSI_LiDAR_add_mean, x2_HSI_LiDAR_add_mean, x3_HSI_LiDAR_add_mean), dim=1)
        output_subtract_mean = torch.cat((x1_HSI_LiDAR_subtract_mean, x2_HSI_LiDAR_subtract_mean, x3_HSI_LiDAR_subtract_mean), dim=1)
        output_add_mean = self.fc_4_HSI_LiDAR(output_add_mean)
        output_subtract_mean = self.fc_5_HSI_LiDAR(output_subtract_mean)

        x4_subtract_mean = abs(x4 - x4_LiDAR)
        x4_add_mean = x4 + x4_LiDAR

        if self.dataset_name == "Augsburg":
            output = self.fc_3_HSI_LiDAR(0.9 * x_tr + 0.1 * x_tr_LiDAR + 0.1 * output_add_mean - 0.2 * output_subtract_mean)
        else:
            output = self.fc_3_HSI_LiDAR(0.8 * x_tr + 0.2 * x_tr_LiDAR + 0.2 * output_add_mean - 0.2 * output_subtract_mean)

        return output
