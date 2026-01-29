import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes, dropout=False, act='relu', output_patch_scores=True, input_dim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self.output_patch_scores = output_patch_scores
        
        if input_dim != 512:
            self._fc1 = [nn.Linear(input_dim, 512)]
            
            if act.lower() == 'relu':
                self._fc1 += [nn.ReLU()]
            elif act.lower() == 'gelu':
                self._fc1 += [nn.GELU()]

            if dropout:
                self._fc1 += [nn.Dropout(0.25)]
                
            self._fc1 = nn.Sequential(*self._fc1)
        else:
            self._fc1 = nn.Identity()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        
        self._fc2 = nn.Linear(512, self.n_classes)
        
        self.patch_classifier = nn.Linear(512, self.n_classes)
        
        self.apply(initialize_weights)

    def forward(self, x, return_attn=False):
        h = x.float()
        
        h = self._fc1(h)
        
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
            
        original_n_patches = h.shape[1]
            
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        
        padded_n_patches = h.shape[1]

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)

        h = self.pos_layer(h, _H, _W)
        
        h = self.layer2(h)
        
        h_normalized = self.norm(h)
        
        cls_feature = h_normalized[:, 0]
        
        patch_features = h_normalized[:, 1:]
        
        if add_length > 0:
            patch_features = patch_features[:, :original_n_patches, :]
            
        logits = self._fc2(cls_feature)
        
        if self.output_patch_scores:
            import torch.nn.functional as F
            
            patch_logits = self.patch_classifier(patch_features)
            
            patch_scores = F.softmax(patch_logits, dim=-1)
            
            if B == 1:
                patch_scores_matrix = patch_scores.squeeze(0)
            
            if return_attn:
                return {
                    'bag_logits': logits,
                    'patch_logits': patch_logits,
                    'patch_scores': patch_scores,
                    'patch_scores_matrix': patch_scores_matrix,
                    'patch_features': patch_features
                }, patch_scores_matrix
            else:
                return {
                    'bag_logits': logits,
                    'patch_logits': patch_logits,
                    'patch_scores': patch_scores,
                    'patch_scores_matrix': patch_scores_matrix
                }
        else:
            if return_attn:
                dummy_attn = torch.ones(h.shape[0], original_n_patches, 1) / original_n_patches
                return logits, dummy_attn.to(h.device)
            else:
                return logits

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    
    model = TransMIL(n_classes=10, dropout=False, act='relu', output_patch_scores=True)
    
    outputs = model(data)
    
    print("Bag logits shape:", outputs['bag_logits'].shape)
    print("Patch logits shape:", outputs['patch_logits'].shape)
    print("Patch scores shape:", outputs['patch_scores'].shape)
    print("Patch scores matrix shape:", outputs['patch_scores_matrix'].shape)
    
    row_sums = outputs['patch_scores_matrix'].sum(dim=1)
    print("Row sums (should be all 1.0):", row_sums[:5])
    print("All rows sum to 1:", torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5))