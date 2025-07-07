import torch
from sklearn.metrics import roc_auc_score, f1_score

def compute_mia_score(original_model, unlearned_model, test_loader):
    mia_scores = []
    original_model.eval()
    unlearned_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].cpu().numpy()
            orig_out = original_model(torch.tensor(inputs)).cpu().numpy()
            unlearned_out = unlearned_model(torch.tensor(inputs)).cpu().numpy()
            scores = np.abs(orig_out - unlearned_out).mean(axis=1)
            mia_scores.append(scores.mean())
    return np.mean(mia_scores)


def compute_classification_metrics(preds, labels):
    preds = torch.sigmoid(preds).cpu().numpy()
    labels = labels.cpu().numpy()
    auc = roc_auc_score(labels, preds, multi_class='ovr')
    f1 = f1_score(labels, preds.argmax(axis=1), average='macro')
    return {'AUC': auc, 'F1': f1}