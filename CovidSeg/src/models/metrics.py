from collections import defaultdict

from scipy import spatial
import numpy as np
import torch

from sklearn.metrics import confusion_matrix, classification_report, plot_precision_recall_curve_
from sklearn.metrics import plot_confusion_matrix_withcm, plot_confusion_matrix, roc_curve, auc
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt 
import os
import pandas as pd 
import json
import numpy as np

class SegMonitor:
    def __init__(self):
        self.cf = None
        self.n_samples = 0
        self.y_true = []
        self.y_pred = []

    def val_on_batch(self, model, batch):
        masks = batch["masks"]
        self.n_samples += masks.shape[0]
        pred_mask = model.predict_on_batch(batch)
        ind = masks != 255
        masks = masks[ind]
        pred_mask = pred_mask[ind]
        self.y_true += masks.numpy().tolist()
        self.y_pred += pred_mask.cpu().numpy().tolist()


        dice = dice_coeff(masks.cuda(), pred_mask)

        cpu_p_mask = pred_mask.cpu().detach().numpy()
        cpu_mask = masks.cpu().detach().numpy()

        # i = smp.utils.metrics.IoU()
        # ci = i(pred_mask, masks.cuda()).cpu().detach().numpy()
        # print(ci)

        labels = np.arange(model.n_classes)

        cf = confusion_multi_class(pred_mask.float(), masks.cuda().float(),
                                   labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def get_avg_score(self):
        # return -1
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter

        nz = union != 0
        iou = Inter / np.maximum(union, 1)
        mIoU = np.mean(iou[nz])
        iou[~nz] = np.nan

        val_dict = {'val_score': mIoU}
        val_dict['iou'] = iou

        return val_dict

    def report(self, savedir):
        self.y_true = [x if x < 3 else 0 for x in self.y_true]
        self.y_pred = [x if x < 3 else 0 for x in self.y_pred]
        cm = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        with open(os.path.join(savedir, 'report.json'), 'w') as f:
            json.dump(report, f)
        plot_confusion_matrix_withcm(cm, labels=['Not Lung Nor COVID', 'Lung', 'COVID'], 
            cmap=plt.cm.Blues, values_format = '.4f')
        plt.savefig(os.path.join(savedir, 'cm.png'), bbox_inches='tight')
        plt.close()


class LocMonitor:
    def __init__(self):
        self.cf = None
        self.n_samples = 0

    def val_on_batch(self, model, batch):
        masks = batch["masks"]
        self.n_samples += masks.shape[0]
        pred_mask = model.predict_on_batch(batch)
        ind = masks != 255
        masks = masks[ind]
        pred_mask = pred_mask[ind]

        labels = np.arange(model.n_classes)
        cf = confusion_multi_class(pred_mask.float(), masks.cuda().float(),
                                   labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def get_avg_score(self):
        # return -1
        Inter = np.diag(self.cf)
        G = self.cf.sum(axis=1)
        P = self.cf.sum(axis=0)
        union = G + P - Inter
        nz = union != 0
        iou = Inter / np.maximum(union, 1)
        mIoU = np.mean(iou[nz])
        iou[~nz] = np.nan

        val_dict = {'val_score': mIoU}
        val_dict['iou'] = iou

        return val_dict


def dice_coeff(probs, target):
    """Dice loss.
    :param input: The input (predicted)
    :param target:  The target (ground truth)
    :returns: the Dice score between 0 and 1.
    """
    eps = 0.0001

    iflat = probs.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice

def confusion_multi_class(prediction, truth, labels):
    """
    cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
            y_pred=truth.cpu().numpy().ravel(),
                    labels=labels)
    """
    # nclasses = labels.max() + 1
    nclasses = labels.max() + 1
    cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float,
                      device=prediction.device)
    prediction = prediction.view(-1).long()
    truth = truth.view(-1)
    to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype,
                           device=prediction.device)
    for c in range(nclasses):
        true_mask = (truth == c)
        test = prediction[true_mask].cpu().numpy()
        pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
        cf2[:, c] = pred_one_hot

    # print(cf2)
    cf2 = cf2.cpu().numpy()
    cf1 =  np.zeros((nclasses, nclasses), dtype=np.float)
    cf1[0:3, 0:3] = cf2[0:3, 0:3]
    # return cf2.cpu().numpy()
    return cf1


def confusion_binary_class(prediction, truth):
    confusion_vector = prediction / truth

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    cm = np.array([[tn, fp],
                   [fn, tp]])
    return cm
