import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def test_epoch(epoch, model, test_loader, logger, device, args, etf_head = False, return_result = False, return_feat = False, keep = False):
    mse_loss = nn.MSELoss().to(device)
    model.eval()

    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    feats = []
    middle_feature = [
        [],
        [],
        [],
    ]
    with torch.no_grad():
        for i, (video_feat, label) in enumerate(test_loader):
            # print(video_feat.shape, label.shape)
            video_feat = video_feat.to(device)
            if keep:
                video_feat = video_feat[:,:68,:]
            label = label.float().to(device)
            out = model(video_feat, return_feat)

            if etf_head:
                pred = model.get_score( out['output'] )
            else:
                pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())

            middle_feature[0].append( out['embed'].detach().squeeze().cpu().numpy()  if out['embed'] is not None else None)
            middle_feature[1].append( out['other'] )
            middle_feature[2].append( video_feat.detach().squeeze().cpu().numpy() )

            loss = mse_loss(pred, label)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    # print(preds.shape, labels.shape, etf_head)
    
    avg_coef, _ = spearmanr(preds, labels)
    # print("spearmanr", avg_coef, preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)
    pred_scores = preds.reshape(-1,)
    true_scores = labels.reshape(-1,)
    rl2 = 100 * np.power((pred_scores - true_scores) /
                       (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)
        logger.add_scalar('Test rl2', rl2, epoch)
    if return_result:
        if return_feat:
            return avg_loss, avg_coef, rl2, preds, labels, middle_feature
        return avg_loss, avg_coef, rl2, preds, labels
    # print(preds.tolist())
    # print(labels.tolist())
    return avg_loss, avg_coef, rl2
