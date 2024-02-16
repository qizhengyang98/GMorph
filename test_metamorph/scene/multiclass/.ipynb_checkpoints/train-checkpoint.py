from tabnanny import verbose
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import average_precision_score
import numpy as np

import model_utils
from dataloader import load_data

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def train(model, tr, n_epochs, lr):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.5, verbose=True)

    total_steps = 0
    for e in range(n_epochs):
        for batch, label_logits in tr:
            optimizer.zero_grad()
            
            batch = batch.to(device)
            label_logits = label_logits.to(device)
            
            preds_probs = model(batch)
            loss = loss_fn(preds_probs, label_logits)
                    
            loss.backward()
            optimizer.step()

            total_steps +=1                                
            model_utils.print_training_loss_summary(loss.item(), total_steps, e+1, n_epochs, len(tr))
        scheduler.step()
    torch.save(model.state_dict(), 'objectNet.model')


def test(model, dataloader):
    model.eval()
    pred = None
    target = None

    with torch.no_grad():
        for batch, label_logits in dataloader:
            batch = batch.to(device)
            label_logits = label_logits.to(device)
            
            pred_scores = model(batch)
            preds_probs = torch.sigmoid(pred_scores)

            if target is None:
                target = np.array(label_logits.cpu())
                pred = np.array(preds_probs.cpu())
            else:
                target = np.append(target, np.array(label_logits.cpu()), axis=0)
                pred = np.append(pred, np.array(preds_probs.cpu()), axis=0)
            
        mAP = average_precision_score(target, pred, average='micro')
            
        return mAP


if __name__ == '__main__':
    tr, te = load_data()
    model = model_utils.get_resnet34_model_with_custom_head()
    model.load_state_dict(torch.load('objectNet.model', map_location=device))
    model = model.to(device)
    # train(model, tr, 40, 0.001)
    acc = test(model, te)
    print(acc)