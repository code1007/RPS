import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch
import os
import torch.nn.functional as F

def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def five_scores(bag_labels, bag_predictions, threshold_optimal=None):
    if isinstance(bag_labels[0], str):
        bag_labels = np.array([int(label) for label in bag_labels])
    else:
        bag_labels = np.array(bag_labels)
    
    bag_predictions = np.array(bag_predictions)
    unique_labels = np.unique(bag_labels)
    n_classes = len(unique_labels)
    
    try:
        if n_classes <= 2:  
            if len(bag_predictions.shape) == 1 or bag_predictions.shape[1] == 1:
                auc_value = roc_auc_score(bag_labels, bag_predictions)
            else:
                auc_value = roc_auc_score(bag_labels, bag_predictions[:, 1])
        else:  
            if len(bag_predictions.shape) == 1:
                one_hot_preds = np.zeros((len(bag_predictions), n_classes))
                for i, pred in enumerate(bag_predictions):
                    pred_class = min(int(np.round(pred * (n_classes-1))), n_classes-1)
                    one_hot_preds[i, pred_class] = 1
                auc_value = roc_auc_score(bag_labels, one_hot_preds, multi_class='ovr', average='macro')
            else:
                auc_value = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr', average='macro')
    except Exception as e:
        auc_value = 0.5
    
    if threshold_optimal is None:
        if len(bag_predictions.shape) == 1:
            if n_classes <= 2:
                class_predictions = (bag_predictions > 0.5).astype(int)
            else:
                class_predictions = np.round(bag_predictions * (n_classes-1)).astype(int)
        else:
            class_predictions = np.argmax(bag_predictions, axis=1)
    else:
        class_predictions = (bag_predictions > threshold_optimal).astype(int)
    
    precision, recall, fscore, _ = precision_recall_fscore_support(
        bag_labels, class_predictions, average='macro', zero_division=0
    )
    accuracy = accuracy_score(bag_labels, class_predictions)
    
    return accuracy, auc_value, precision, recall, fscore, threshold_optimal

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def extract_patch_class_scores(model, data_loader, device, output_dir, model_type='transmil'):

    import os
    import torch
    import torch.nn.functional as F

    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if isinstance(data[0], (list, tuple)):
                bag = [item.to(device) for item in data[0]]
            else:
                bag = data[0].to(device)
            
            labels = data[1].to(device)
            slide_ids = data[2] if len(data) > 2 else [f"slide_{i}_{j}" for j in range(len(labels))]
            
            if model_type.lower() == 'transmil':
                outputs = model(bag)
                
                if isinstance(outputs, dict) and 'patch_scores_matrix' in outputs:
                    patch_scores = outputs['patch_scores_matrix']  # [patches, classes]
                    bag_logits = outputs['bag_logits']  # [batch, classes]
                else:
                    continue
            
            elif model_type.lower() == 'rps':
                logits = model.forward_test(bag)
                
                x = model.patch_to_emb(bag)
                x = model.dp(x)
                
                patch_logits = []
                for i in range(x.size(1)):
                    patch_feat = model.online_encoder(x[:, i:i+1, :])
                    patch_logit = model.predictor(patch_feat)
                    patch_logits.append(patch_logit)
                
                patch_logits = torch.stack(patch_logits, dim=1)  # [batch, patches, classes]
                
                patch_scores = F.softmax(patch_logits, dim=-1)
                
                if patch_scores.size(0) == 1:
                    patch_scores = patch_scores.squeeze(0)  # [patches, classes]
                
                bag_logits = logits
            
            else:
                continue
            

            for j in range(len(labels)):
                if isinstance(slide_ids, list):
                    slide_id = slide_ids[j]
                else:
                    slide_id = slide_ids[j].item() if hasattr(slide_ids[j], 'item') else slide_ids[j]
                
                label = labels[j].item() if hasattr(labels[j], 'item') else labels[j]
                
                if len(patch_scores.shape) == 3:  # [batch, patches, classes]
                    current_patch_scores = patch_scores[j]
                else:  # [patches, classes]
                    current_patch_scores = patch_scores
                
                if len(bag_logits.shape) == 2:  # [batch, classes]
                    current_bag_logits = bag_logits[j]
                else:  # [classes]
                    current_bag_logits = bag_logits
                
                output_file = os.path.join(output_dir, f"slide_{slide_id}_patch_scores.pt")
                torch.save({
                    'slide_id': slide_id,
                    'label': label,
                    'patch_scores': current_patch_scores.cpu(),  # [patches, classes]
                    'bag_logits': current_bag_logits.cpu(),  # [classes]
                    'patch_count': current_patch_scores.size(0)
                }, output_file)
                
