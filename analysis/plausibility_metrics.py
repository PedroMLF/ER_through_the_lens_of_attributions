import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def compute_auc_score(labels, attributions):
    return roc_auc_score(labels, attributions)

def compute_average_precision(labels, attributions):
    return average_precision_score(labels, attributions)

def compute_recall_at_k(labels, attributions):
    # Get the ranks of the top-k attributions (k = number of annotated tokens)
    idxs = torch.argsort(attributions, descending=True)[:sum(labels)]
    # Count how many of the top-k attribution ranks are lower than k and divide by k
    metric_scores = len([idx for idx in idxs if labels[idx] == 1]) / sum(labels)
    if isinstance(metric_scores, torch.Tensor):
        metric_scores = metric_scores.item()
    return metric_scores
