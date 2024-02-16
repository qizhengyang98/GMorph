import torch
import numpy as np
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

import os, sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def test_top1(model, data_loader, device, multi_idx=None):
    correct_1 = 0.0
    model.eval()

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(data_loader):

            image = image.to(device)
            label = label.to(device)

            if multi_idx is not None:
                output = model(image)[multi_idx]
            else:
                output = model(image)
            _, pred = output.topk(1, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top1
            correct_1 += correct[:, :1].sum()

    top1_acc = (correct_1 / len(data_loader.dataset)).item()
    print("Top 1 accuracy: ", top1_acc)

    return top1_acc

def test_top5(model, data_loader, device, multi_idx=None):
    correct_5 = 0.0
    model.eval()

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(data_loader):

            image = image.to(device)
            label = label.to(device)

            if multi_idx is not None:
                output = model(image)[multi_idx]
            else:
                output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            correct_5 += correct[:, :5].sum()

    top5_acc = (correct_5 / len(data_loader.dataset)).item()
    print("Top 5 accuracy: ", top5_acc)

    return top5_acc

def test_multiclass(model, dataloader, device, multi_idx=None):
    model.eval()
    pred = None
    target = None

    with torch.no_grad():
        for batch, label_logits in dataloader:
            batch = batch.to(device)
            label_logits = label_logits.to(device)
            
            if multi_idx is not None:
                pred_scores = model(batch)[multi_idx]
            else:
                pred_scores = model(batch)
            preds_probs = torch.sigmoid(pred_scores)

            if target is None:
                target = np.array(label_logits.cpu())
                pred = np.array(preds_probs.cpu())
            else:
                target = np.append(target, np.array(label_logits.cpu()), axis=0)
                pred = np.append(pred, np.array(preds_probs.cpu()), axis=0)
            
        mAP = average_precision_score(target, pred, average='micro')
        print('\nTest: mean Average Precision: {}\n'.format(mAP))
            
        return mAP

def test_salient(model, dataloader, device, multi_idx=None):
    model.eval()
    pred = None
    target = None

    with torch.no_grad():
        for batch, label_logits in dataloader:
            batch = batch.to(device)
            label_logits = F.one_hot(label_logits.reshape(-1).to(torch.int64), 5).to(device)
            
            if multi_idx is not None:
                pred_scores = model(batch)[multi_idx]
            else:
                pred_scores = model(batch)
            preds_probs = torch.sigmoid(pred_scores)

            if target is None:
                target = np.array(label_logits.cpu())
                pred = np.array(preds_probs.cpu())
            else:
                target = np.append(target, np.array(label_logits.cpu()), axis=0)
                pred = np.append(pred, np.array(preds_probs.cpu()), axis=0)
            
        mAP = average_precision_score(target, pred, average='macro')
        print('\nTest: mean Average Precision: {}\n'.format(mAP))
            
        return mAP


def test_multi_result(test_loader_list, model, device):
    results = []
    for i in range(len(test_loader_list)):
        results.append(0)

    model.to(device).eval()

    with torch.no_grad():
        with HiddenPrints():
            for i, test_loader in enumerate(test_loader_list):
                if i == 0:
                    results[i] = test_top1(model, test_loader, device, multi_idx=i)
                elif i == 1:
                    results[i] = test_multiclass(model, test_loader, device, multi_idx=i)
                elif i == 2:
                    results[i] = test_salient(model, test_loader, device, multi_idx=i)
                else:
                    raise NotImplementedError

    str = ''
    for i in range(len(results)):
        str += f"net{i+1} Result: {results[i]*100}%   "
    print(str + "\n")

    return torch.tensor(results)

def test_multi_scene_multiclass(test_loader_list, model, device):
    results = []
    for i in range(len(test_loader_list)):
        results.append(0)

    model.to(device).eval()

    with torch.no_grad():
        with HiddenPrints():
            for i, test_loader in enumerate(test_loader_list):
                if i == 0:
                    results[i] = test_top1(model, test_loader, device, multi_idx=i)
                elif i == 1:
                    results[i] = test_multiclass(model, test_loader, device, multi_idx=i)
                else:
                    raise NotImplementedError

    str = ''
    for i in range(len(results)):
        str += f"net{i+1} Result: {results[i]*100}%   "
    print(str + "\n")

    return torch.tensor(results)

def test_multi_multiclass_salient(test_loader_list, model, device):
    results = []
    for i in range(len(test_loader_list)):
        results.append(0)

    model.to(device).eval()

    with torch.no_grad():
        with HiddenPrints():
            for i, test_loader in enumerate(test_loader_list):
                if i == 0:
                    results[i] = test_multiclass(model, test_loader, device, multi_idx=i)
                elif i == 1:
                    results[i] = test_salient(model, test_loader, device, multi_idx=i)
                else:
                    raise NotImplementedError

    str = ''
    for i in range(len(results)):
        str += f"net{i+1} Result: {results[i]*100}%   "
    print(str + "\n")

    return torch.tensor(results)

def test_multi_scene_salient(test_loader_list, model, device):
    results = []
    for i in range(len(test_loader_list)):
        results.append(0)

    model.to(device).eval()

    with torch.no_grad():
        with HiddenPrints():
            for i, test_loader in enumerate(test_loader_list):
                if i == 0:
                    results[i] = test_top1(model, test_loader, device, multi_idx=i)
                elif i == 1:
                    results[i] = test_salient(model, test_loader, device, multi_idx=i)
                else:
                    raise NotImplementedError

    str = ''
    for i in range(len(results)):
        str += f"net{i+1} Result: {results[i]*100}%   "
    print(str + "\n")

    return torch.tensor(results)