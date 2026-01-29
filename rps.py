import torch
import numpy as np 
from torch import nn
from modules.datten import *
import torch.nn.functional as F
from modules.satten import *

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=1.,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss
        
class RPS(nn.Module):
    def __init__(self, mlp_dim=512, n_classes=10, dropout=0.25, act='relu', da_act='gelu', 
                 k_top_patches=0.3, top_k=3, baseline='selfattn', head=8, 
                 temp_t=0.1, temp_s=1.0, mrh_sche=None, attn_layer=0, select_mask=True,
                 use_text_guidance=True, text_features_path=None, use_cosine_similarity=True):
        super(RPS, self).__init__()
        
        self.mrh_sche = mrh_sche
        # self.attn_layer = attn_layer
        self.select_mask = select_mask
        self.k_top_patches = k_top_patches
        self.top_k = top_k
        self.baseline = baseline
        self.use_text_guidance = use_text_guidance

        self.patch_to_emb = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim,head=head)
        elif baseline == 'attn':
            self.online_encoder = DAttention(mlp_dim,da_act)
        elif baseline == 'dsmil':
            self.online_encoder = DSMIL(mlp_dim=mlp_dim,mask_ratio=mask_ratio)
        elif baseline == 'transmil':
            from modules.transmil import TransMIL
            self.online_encoder = TransMIL(n_classes=n_classes, dropout=dropout>0, act=act, output_patch_scores=True, input_dim=512)

        self.predictor = nn.Linear(mlp_dim, n_classes)
        
        if self.use_text_guidance:
            from modules.text_guided_image_decoder import TextGuidedImageDecoder
            
            class Config:
                def __init__(self):
                    self.input_size = 512  
                    self.hidden_size = 256  
                    self.text_dim = 512    
            
            config = Config()
            self.text_guided_image_decoder = TextGuidedImageDecoder(config, num_classes=n_classes)
            
            if text_features_path:
                text_data = torch.load(text_features_path)
                if isinstance(text_data, dict):
                    if 'features' in text_data:
                        text_features = text_data['features']
                    elif 'text_features' in text_data:
                        text_features = text_data['text_features']
                    elif 'embeddings' in text_data:
                        text_features = text_data['embeddings']
                    else:
                        text_features = next(iter(text_data.values()))
                        print(f"Warning: Using first tensor from dict with keys: {list(text_data.keys())}")
                else:
                    text_features = text_data
                self.register_buffer('text_features', text_features)
            else:
                default_path = './text_features.pt'
                try:
                    text_data = torch.load(default_path)
                    if isinstance(text_data, dict):
                        if 'features' in text_data:
                            text_features = text_data['features']
                        elif 'text_features' in text_data:
                            text_features = text_data['text_features']
                        elif 'embeddings' in text_data:
                            text_features = text_data['embeddings']
                        else:
                            text_features = next(iter(text_data.values()))
                            print(f"Warning: Using first tensor from dict with keys: {list(text_data.keys())}")
                    else:
                        text_features = text_data
                    self.register_buffer('text_features', text_features)
                except:
                    print(f"Warning: Could not load text features from {default_path}")
                    self.register_buffer('text_features', torch.randn(n_classes, 512))
            
            self.text_guided_classifier = nn.Linear(mlp_dim, n_classes)
        
        self.apply(initialize_weights)
        self.temp_t = temp_t
        self.temp_s = temp_s
        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)
        # self.predictor_cl = nn.Identity()
        #self.target_predictor = nn.Identity()
 

    def select_patch_mask(self, score_matrix, num_patches, num_classes, k_top_patches=0.3, top_k=3, save_top_patches=True):
        if score_matrix is None or score_matrix.numel() == 0:
            print("Warning: Empty score matrix provided to select_patch_mask")
            if save_top_patches:
                return torch.tensor([], dtype=torch.long), {}
            return torch.tensor([], dtype=torch.long)
        
        score_matrix = score_matrix.cpu()
        
        if len(score_matrix.shape) == 2:
            patch_scores = score_matrix  # [num_patches, num_classes]
            batch_size = 1
        elif len(score_matrix.shape) == 3:
            batch_size = score_matrix.size(0)
        else:
            raise ValueError(f"Unsupported score_matrix shape: {score_matrix.shape}")
        
        min_patches = max(int(num_patches * 0.05), 100)  
        unique_indices = set()
        
        top_patches_dict = {} if save_top_patches else None

        for b in range(batch_size):
            try:
                if len(score_matrix.shape) == 3:
                    patch_scores = score_matrix[b]  # [num_patches, num_classes]
                topk_per_patch = torch.topk(patch_scores, k=min(top_k, num_classes), dim=1)[1]  # [num_patches, top_k]
                class_frequency = torch.zeros(num_classes, dtype=torch.long)
                for patch_idx in range(topk_per_patch.size(0)):
                    for class_idx in topk_per_patch[patch_idx]:
                        class_frequency[class_idx] += 1
                candidate_classes = torch.topk(class_frequency, k=min(3, num_classes))[1]
                
                for rank, class_idx in enumerate(candidate_classes):
                    class_scores = patch_scores[:, class_idx]
                    num_select = max(1, int(num_patches * k_top_patches))
                    top_patches = torch.topk(class_scores, k=min(num_select, num_patches))[1]
                    
                    if save_top_patches:
                        class_name = f"Class_{class_idx.item()}"
                        top_patches_dict[class_name] = {
                            'patch_indices': top_patches.tolist()
                        }
                    
                    for idx in top_patches.tolist():
                        unique_indices.add(idx)
            except Exception as e:
                print(f"Error processing batch {b}: {e}")
                continue
        
        if len(unique_indices) == 0:
            print("Warning: No indices selected, using default")
            default_count = max(1, int(0.1 * num_patches))
            selected_indices = torch.arange(min(default_count, num_patches), dtype=torch.long)
            if save_top_patches:
                return selected_indices, {}
            return selected_indices
        
        selected_indices = torch.tensor(sorted(unique_indices), dtype=torch.long)
        
        if save_top_patches:
            return selected_indices, top_patches_dict
        return selected_indices

    @torch.no_grad()
    def forward_teacher(self, x, score_matrix=None, k_top_patches=0.3, top_k=3, save_top_patches=True):
        x = self.patch_to_emb(x)
        x = self.dp(x)
    
        num_patches = x.size(1)
        num_classes = self.predictor.out_features
    
        k_top_patches = self.k_top_patches if k_top_patches is None else k_top_patches
        top_k = self.top_k if top_k is None else top_k
    
        selected_patch_indices = None  
    
        if score_matrix is not None:
            try:
                result = self.select_patch_mask(score_matrix, 
                                              num_patches, 
                                              num_classes, 
                                              k_top_patches=self.k_top_patches,
                                              top_k=self.top_k)  
                
                if isinstance(result, tuple):
                    keep_indices, top_patches_dict = result
                else:
                    keep_indices = result
                    
                selected_patch_indices = keep_indices.clone()
                
                if keep_indices.numel() > 0:
                    x = x[:, keep_indices]
                else:
                    print("Warning: No patches selected, using all patches")
            except Exception as e:
                print(f"Error selecting patches: {e}")
        
        if self.baseline == 'dsmil':
            _, x, attn = self.online_encoder(x, return_attn=True)
        else:
            result = self.online_encoder(x, return_attn=True)
            if isinstance(result, tuple) and len(result) == 2:
                x_or_dict, attn = result
                if isinstance(x_or_dict, dict):
                    x = x_or_dict['bag_logits']
                else:
                    x = x_or_dict
            else:
                x = result
                attn = None
        
        if attn is not None:
            score_matrix = attn  # [batch_size, num_patches, num_classes] 或 [batch_size, num_patches]
        else:
            patch_features = self.patch_to_emb(x.unsqueeze(0) if len(x.shape) == 1 else x)
            if len(patch_features.shape) == 3:
                # [batch_size, num_patches, feature_dim]
                batch_size, num_patches, feature_dim = patch_features.shape
                patch_features_flat = patch_features.view(-1, feature_dim)  # [batch_size*num_patches, feature_dim]
                patch_scores = self.predictor(patch_features_flat)  # [batch_size*num_patches, num_classes]
                score_matrix = patch_scores.view(batch_size, num_patches, -1)  # [batch_size, num_patches, num_classes]
            else:
                score_matrix = None
        
        return x, attn, selected_patch_indices, score_matrix  

    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        
        score_matrix = None

        if self.baseline == 'transmil':
            transmil_outputs = self.online_encoder(x)
            
            if isinstance(transmil_outputs, dict):
                student_logit = transmil_outputs['bag_logits']
                
                score_matrix = transmil_outputs.get('patch_scores_matrix', None)
                
                if self.use_text_guidance and hasattr(self, 'text_features'):
                    patch_features = transmil_outputs.get('patch_features', None) 
                    
                    if patch_features is not None:
                        if len(patch_features.shape) == 3:
                            patch_features_2d = patch_features.squeeze(0)
                        else:
                            patch_features_2d = patch_features
                        
                        text_guided_features = self.text_guided_image_decoder(patch_features_2d, self.text_features)
                        text_guided_logit = self.text_guided_classifier(text_guided_features)
                        
                        alpha = 0.7
                        beta = 0.3
                        final_logit = alpha * student_logit + beta * text_guided_logit
                        student_logit = final_logit
                
                if return_attn:
                    attn = transmil_outputs.get('patch_scores_matrix', None)
                    return student_logit, attn, None, score_matrix  
                else:
                    return student_logit, None, None, score_matrix  
            else:
                x = transmil_outputs
        else:
            if return_attn:
                x, a = self.online_encoder(x, return_attn=True, no_norm=no_norm)
            else:
                x = self.online_encoder(x)

            if self.baseline == 'dsmil':
                pass
            else:   
                x = self.predictor(x)
            
            if hasattr(self, 'predictor'):
                if len(x.shape) == 2:  # [num_patches, num_classes]
                    score_matrix = x.unsqueeze(0)  # [1, num_patches, num_classes]
                elif len(x.shape) == 3:  # [batch_size, num_patches, num_classes]
                    score_matrix = x
            
            if self.use_text_guidance and hasattr(self, 'text_features'):
                text_guided_features = self.text_guided_image_decoder(x, self.text_features)
                text_guided_logit = self.text_guided_classifier(text_guided_features)
                
                alpha = 0.7
                beta = 0.3
                final_logit = alpha * x + beta * text_guided_logit
                x = final_logit

            if return_attn:
                return x, a, None, score_matrix  
            else:
                return x, None, None, score_matrix  

    def pure(self, x):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.baseline == 'dsmil':
            x, _ = self.online_encoder(x)
        else:
            x = self.online_encoder(x)
            x = self.predictor(x)

        if self.training:
            return x, 0, ps, ps
        else:
            return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss

    def forward(self, x, attn=None, teacher_cls_feat=None, i=None, score_matrix=None, k_top_patches=0.3, top_k=3, save_top_patches=True):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)
    
        top_patches_dict = {}
    
        if score_matrix is not None and self.select_mask:
            num_patches = x.size(1)
            num_classes = self.predictor.out_features
        
            result = self.select_patch_mask(score_matrix, 
                                          num_patches, 
                                          num_classes, 
                                          k_top_patches=self.k_top_patches,
                                          top_k=self.top_k,
                                          save_top_patches=save_top_patches)  
        
            if isinstance(result, tuple):
                keep_indices, top_patches_dict = result
            else:
                keep_indices = result
                
            mask_ids = torch.arange(num_patches, device=x.device).unsqueeze(0)
            mask_ids = torch.tensor(list(set(range(num_patches)) - set(keep_indices.cpu().numpy())), 
                                    dtype=torch.long, device=x.device).unsqueeze(0)
            len_keep = keep_indices.size(0)
        else:
            len_keep = ps
            mask_ids = None

        if self.baseline == 'transmil':
            transmil_outputs = self.online_encoder(x)
            
            if isinstance(transmil_outputs, dict):
                student_logit = transmil_outputs['bag_logits']
                
                if self.use_text_guidance and hasattr(self, 'text_features'):
                    patch_features = transmil_outputs.get('patch_features', None)
                    
                    if patch_features is not None:
                        if len(patch_features.shape) == 3:
                            patch_features_2d = patch_features.squeeze(0)  # [N, 512]
                        else:
                            patch_features_2d = patch_features
                        
                        text_guided_features = self.text_guided_image_decoder(patch_features_2d, self.text_features)
                        
                        text_guided_logit = self.text_guided_classifier(text_guided_features)
                        
                        alpha = 0.7  
                        beta = 0.3  
                        final_logit = alpha * student_logit + beta * text_guided_logit
                        
                        student_logit = final_logit
                
                if teacher_cls_feat is not None:
                    cls_loss = self.forward_loss(student_cls_feat=student_logit, teacher_cls_feat=teacher_cls_feat)
                else:
                    cls_loss = 0.
                
                return student_logit, cls_loss, ps, len_keep, top_patches_dict
            else:
                student_logit = transmil_outputs
                cls_loss = self.forward_loss(student_cls_feat=student_logit, teacher_cls_feat=teacher_cls_feat) if teacher_cls_feat is not None else 0.
                return student_logit, cls_loss, ps, len_keep
                
        elif self.baseline == 'dsmil':
            # forward online network
            student_logit, student_cls_feat = self.online_encoder(x, len_keep=len_keep, mask_ids=mask_ids, mask_enable=True)
            
            if self.use_text_guidance and hasattr(self, 'text_features'):
                if self.use_cosine_similarity:
                    slide_features, similarity_scores = self.text_guided_image_decoder(
                        student_cls_feat, self.text_features, return_similarity=True
                    )
                    student_logit = similarity_scores
                else:
                    text_guided_features = self.text_guided_image_decoder(student_cls_feat, self.text_features)
                    text_guided_logit = self.text_guided_classifier(text_guided_features)
                    
                    alpha = 0.7
                    beta = 0.3
                    final_logit = alpha * student_logit + beta * text_guided_logit
                    student_logit = final_logit

            # cl loss
            cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss, ps, len_keep, top_patches_dict
        else:
            # forward online network 
            student_cls_feat = self.online_encoder(x, len_keep=len_keep, mask_ids=mask_ids, mask_enable=True)

            # prediction
            student_logit = self.predictor(student_cls_feat)
            
            if self.use_text_guidance and hasattr(self, 'text_features'):
                if self.use_cosine_similarity:
                    slide_features, similarity_scores = self.text_guided_image_decoder(
                        student_cls_feat, self.text_features, return_similarity=True
                    )
                    student_logit = similarity_scores
                else:
                    text_guided_features = self.text_guided_image_decoder(student_cls_feat, self.text_features)
                    text_guided_logit = self.text_guided_classifier(text_guided_features)
                    
                    alpha = 0.7
                    beta = 0.3
                    final_logit = alpha * student_logit + beta * text_guided_logit
                    student_logit = final_logit

            # cl loss
            cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss, ps, len_keep, top_patches_dict

def forward_features(self, x):
    if self.baseline == 'transmil':
        transmil_outputs = self.transmil_model(x, return_attn=True)
        bag_logits = transmil_outputs['bag_logits']
        return bag_logits  
    elif self.baseline == 'selfattn':
        features = self.selfattn_model.forward_features(x)
        return features
    else:
        return self.baseline_model.forward_features(x)

def get_features(self, x):
    return self.forward_features(x)