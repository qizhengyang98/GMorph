import torch
import numpy as np
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import transformers
import evaluate

import os, sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def test_multiclass(model, dataloader, device, multi_idx=None):
    model.to(device).eval()
    mAP = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            label_logits = batch['labels'].to(device)
            
            if multi_idx is not None:
                pred_scores = model(inputs)[multi_idx]
            else:
                pred_scores = model(inputs)
            pred_scores = pred_scores if isinstance(pred_scores, torch.Tensor) else pred_scores.logits
            preds_probs = torch.sigmoid(pred_scores)

            mAP += average_precision_score(label_logits.cpu().numpy(), preds_probs.cpu().numpy(), average='micro')
        
        mAP /= len(dataloader)
        print(f'Test: mean Average Precision: {round(mAP*100, 2)}%\n')
        return mAP

def test_salient(model, dataloader, device, multi_idx=None):
    model.to(device).eval()
    pred = None
    target = None

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].reshape(-1).to(device)
            label_logits = F.one_hot(labels.to(torch.int64), 5).to(device)
            
            if multi_idx is not None:
                pred_scores = model(inputs)[multi_idx]
            else:
                pred_scores = model(inputs)
            pred_scores = pred_scores if isinstance(pred_scores, torch.Tensor) else pred_scores.logits
            preds_probs = torch.sigmoid(pred_scores)

            if target is None:
                target = np.array(label_logits.cpu())
                pred = np.array(preds_probs.cpu())
            else:
                target = np.append(target, np.array(label_logits.cpu()), axis=0)
                pred = np.append(pred, np.array(preds_probs.cpu()), axis=0)
            
        mAP = average_precision_score(target, pred, average='macro')
        print(f'Test: mean Average Precision: {round(mAP*100, 2)}%\n')
        return mAP

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
        str += f"net{i+1} Result: {round(results[i]*100, 2)}%   "
    print(str + "\n")

    return torch.tensor(results)

def test_cola(model, dataloader, device, multi_idx=None):
    model.to(device).eval()
    metric = evaluate.load("glue", "cola") # Matthews correlation coefficient
    predictions = None
    targets = None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
            label_logits = batch['labels'].to(device)
            
            if multi_idx is not None:
                pred_scores = model(inputs)[multi_idx]
            else:
                if isinstance(model, transformers.models.bert.modeling_bert.BertForSequenceClassification):
                    pred_scores = model(**inputs)
                else:
                    pred_scores = model(inputs)
            pred_scores = pred_scores if isinstance(pred_scores, torch.Tensor) else pred_scores.logits
            preds = pred_scores.argmax(-1)

            if targets is None:
                targets = np.array(label_logits.cpu())
                predictions = np.array(preds.cpu())
            else:
                targets = np.append(targets, np.array(label_logits.cpu()), axis=0)
                predictions = np.append(predictions, np.array(preds.cpu()), axis=0)

        result = metric.compute(predictions=predictions, references=targets)['matthews_correlation']
        print(f'Test: matthews correlation: {round(result*100, 2)}\n')
        return result

def test_sst2(model, dataloader, device, multi_idx=None):
    model.to(device).eval()
    metric = evaluate.load("glue", "sst2") # Matthews correlation coefficient
    predictions = None
    targets = None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
            label_logits = batch['labels'].to(device)
            
            if multi_idx is not None:
                pred_scores = model(inputs)[multi_idx]
            else:
                if isinstance(model, transformers.models.bert.modeling_bert.BertForSequenceClassification):
                    pred_scores = model(**inputs)
                else:
                    pred_scores = model(inputs)
            pred_scores = pred_scores if isinstance(pred_scores, torch.Tensor) else pred_scores.logits
            preds = pred_scores.argmax(-1)

            if targets is None:
                targets = np.array(label_logits.cpu())
                predictions = np.array(preds.cpu())
            else:
                targets = np.append(targets, np.array(label_logits.cpu()), axis=0)
                predictions = np.append(predictions, np.array(preds.cpu()), axis=0)

        result = metric.compute(predictions=predictions, references=targets)['accuracy']
        print(f'Test: accuracy: {round(result*100, 2)}%\n')
        return result

def test_multi_cola_sst2(test_loader_list, model, device):
    results = []
    for i in range(len(test_loader_list)):
        results.append(0)

    model.to(device).eval()

    with torch.no_grad():
        with HiddenPrints():
            for i, test_loader in enumerate(test_loader_list):
                if i == 0:
                    results[i] = test_cola(model, test_loader, device, multi_idx=i)
                elif i == 1:
                    results[i] = test_sst2(model, test_loader, device, multi_idx=i)
                else:
                    raise NotImplementedError

    str = ''
    for i in range(len(results)):
        str += f"net{i+1} Result: {round(results[i]*100, 2)}%   "
    print(str + "\n")

    return torch.tensor(results)