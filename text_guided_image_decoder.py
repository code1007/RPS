import torch
import torch.nn as nn
from torch.nn import functional as F
from .model_utils import MultiheadAttention

class TextGuidedImageDecoder(nn.Module):
    def __init__(self, config, num_classes=10):
        super(TextGuidedImageDecoder, self).__init__()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        
        self.patch_projection = nn.Linear(10, config.input_size)
        
        self.text_projection = nn.Linear(config.text_dim, config.input_size) if hasattr(config, 'text_dim') and config.text_dim != config.input_size else nn.Identity()
        
        self.norm = nn.LayerNorm(config.input_size)
        
        self.cross_attention = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        
        self.output_projection = nn.Linear(config.input_size, config.input_size)

    def cosine_similarity_classification(self, slide_features, text_features):
        slide_features_norm = F.normalize(slide_features, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)
        
        similarity_scores = torch.mm(slide_features_norm, text_features_norm.T)
        
        return similarity_scores

    def forward(self, patch_features, text_features, return_similarity=False):
        patch_features = self.patch_projection(patch_features)
        
        if text_features.dim() == 2 and text_features.size(1) != self.L:
            text_features = self.text_projection(text_features)
        
        attended_features_list = []
        
        for i in range(self.num_classes):
            query_i = text_features[i:i+1]
            
            query_i = query_i.unsqueeze(0)
            key_value = patch_features.unsqueeze(1)
            
            attended_feature, _ = self.cross_attention(
                query=query_i,
                key=key_value,
                value=key_value,
                chunk_size=64
            )
            attended_features_list.append(attended_feature.squeeze(0))
        
        text_guided_features = torch.cat(attended_features_list, dim=0)
        
        text_guided_features = self.norm(text_guided_features + text_features)
        
        class_weights = F.softmax(torch.sum(text_guided_features, dim=1), dim=0)
        
        weighted_text_features = torch.sum(text_guided_features * class_weights.unsqueeze(1), dim=0, keepdim=True)
        
        H = patch_features.float()
        
        text_guided_attention = torch.mm(H, weighted_text_features.T)
        text_guided_attention = F.softmax(text_guided_attention, dim=0)
        
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = F.softmax(A, dim=0)
        
        combined_attention = 0.7 * A + 0.3 * text_guided_attention
        combined_attention = F.softmax(combined_attention, dim=0)
        
        slide_features = torch.sum(H * combined_attention, dim=0, keepdim=True)
        
        slide_features = self.output_projection(slide_features)
        
        if return_similarity:
            similarity_scores = self.cosine_similarity_classification(slide_features, text_features)
            return slide_features, similarity_scores
        
        return slide_features