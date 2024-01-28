
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
def myMetric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    # Binarize predictions and targets with threshold
    y_pred_bin = (y_pred >= threshold).float()
    y_bin = (y >= threshold).float()

    # Calculate accuracy
    correct = (y_pred_bin == y_bin).float().sum()
    acc = correct / y_bin.numel()

    # Calculate AUROC
    # Note: AUROC is not directly available in torch, and its calculation is non-trivial.
    # You might need to use sklearn or write a custom function for an exact AUROC computation.
    # Here, we provide a placeholder value.
    auroc = torch.tensor(0.0)  # Placeholder for AUROC

    # Calculate precision and recall for each class
    true_positive = (y_pred_bin * y_bin).sum(dim=0)
    false_positive = (y_pred_bin * (1 - y_bin)).sum(dim=0)
    false_negative = ((1 - y_pred_bin) * y_bin).sum(dim=0)

    precision_0 = true_positive[0] / (true_positive[0] + false_positive[0])
    precision_1 = true_positive[1] / (true_positive[1] + false_positive[1])
    recall_0 = true_positive[0] / (true_positive[0] + false_negative[0])
    recall_1 = true_positive[1] / (true_positive[1] + false_negative[1])

    # Calculate F1 score for each class
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

    return {"BCEloss": loss.item(),
            "acc": acc.item(), "auroc": auroc.item(), "precision_1": precision_1.item(),
            "recall_1": recall_1.item(), "f1_1": f1_1.item(), "precision_0": precision_0.item(),
            "recall_0": recall_0.item(), "f1_0": f1_0.item()}

def affinity_metrics(affinity_pred, affinity):
    # Calculate Pearson correlation coefficient
    vx = affinity_pred - torch.mean(affinity_pred)
    vy = affinity - torch.mean(affinity)
    pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    # Calculate RMSE
    mse = F.mse_loss(affinity_pred, affinity, reduction='mean')
    rmse = torch.sqrt(mse)

    return {"pearson": pearson.item(), "rmse": rmse.item()}
    

def print_metrics(metrics):
    out_list = []
    for key in metrics:
        out_list.append(f"{key}:{metrics[key]:6.3f}")
    out = ", ".join(out_list)
    return out


def compute_individual_metrics(pdb_list, inputFile_list, y_list):
    r_ = []
    for i in range(len(pdb_list)):
        pdb = pdb_list[i]
        # inputFile = f"{pre}/input/{pdb}.pt"
        inputFile = inputFile_list[i]
        y = y_list[i]
        (coords, y_pred, protein_nodes_xyz, 
         compound_pair_dis_constraint, pdb, sdf_fileName, mol2_fileName, pre) = torch.load(inputFile)
        result = myMetric(torch.tensor(y_pred).reshape(-1), y.reshape(-1))
        for key in result:
            result[key] = float(result[key])
        result['idx'] = i
        result['pdb'] = pdb
        result['p_length'] = protein_nodes_xyz.shape[0]
        result['c_length'] = coords.shape[0]
        result['y_length'] = y.reshape(-1).shape[0]
        result['num_contact'] = int(y.sum())
        r_.append(result)
    result = pd.DataFrame(r_)
    return result

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report