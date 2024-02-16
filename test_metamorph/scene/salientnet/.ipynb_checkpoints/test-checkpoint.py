import torch
from torch.autograd import Variable
from model import get_resnet18_model_with_custom_head
from dataloader import load_data
import numpy as np
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test(model, test_loader, device):
    accuracy = average_meter()

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = Variable(data,volatile=True).to(device), Variable(target,volatile=True).reshape(-1).to(device)
            output = model(data)

            # print(output.shape, target.shape)

            pred = output.data.max(1)[1]
            prec = pred.eq(target.data).cpu().sum()
            accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accuracy.sum), len(test_loader.dataset), 100. * accuracy.avg))

    return accuracy.avg

def test_map(model, dataloader, device):
    model.eval()
    pred = None
    target = None

    with torch.no_grad():
        for batch, label_logits in dataloader:
            batch = batch.to(device)
            label_logits = F.one_hot(label_logits.reshape(-1).to(torch.int64), 5).to(device)
            
            pred_scores = model(batch)
            preds_probs = torch.sigmoid(pred_scores)

            if target is None:
                target = np.array(label_logits.cpu())
                pred = np.array(preds_probs.cpu())
            else:
                target = np.append(target, np.array(label_logits.cpu()), axis=0)
                pred = np.append(pred, np.array(preds_probs.cpu()), axis=0)
            
        # mAP = average_precision_score(target, pred, average='micro')
        mAP = average_precision_score(target, pred, average='macro')
        print('\nTest: mean Average Precision: {}\n'.format(mAP))
            
        return mAP


if __name__ == '__main__':
    model = get_resnet18_model_with_custom_head(Device=device)
    model.load_state_dict(torch.load('salientNet.model', map_location=device))
    model = model.to(device)
    print(model)

    tr, te = load_data()
    test(model, te, device)
    test_map(model, te, device)