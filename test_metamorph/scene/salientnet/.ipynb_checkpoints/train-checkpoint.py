import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import copy

from model import get_resnet18_model_with_custom_head
from dataloader import load_data

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


model = get_resnet18_model_with_custom_head(Device=device)
tr, te = load_data()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.5, verbose=True)

def train(epoch):
    model.train()
    losses = average_meter()
    accuracy = average_meter()

    for batch_idx, (data, target) in enumerate(tr):
        data, target = Variable(data).to(device), Variable(target).reshape(-1).type(torch.LongTensor).to(device)
        output = model(data)
        loss = loss_fn(output, target)
        losses.update(loss.item(), data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 30 == 0:
            print('Train Epoch: {}\t'
                  'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'                     
                  'Loss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(tr.dataset),
                      100. * batch_idx / len(tr), losses.val))
            print('Training accuracy:', accuracy.val )

def test():
    losses = average_meter()
    accuracy = average_meter()

    model.eval()

    with torch.no_grad():
        for data, target in te:

            data, target = Variable(data,volatile=True).to(device), Variable(target,volatile=True).reshape(-1).type(torch.LongTensor).to(device)
            output = model(data)

            loss = loss_fn(output, target)
            losses.update(loss.item(), data.size(0))

            pred = output.data.max(1)[1]
            prec = pred.eq(target.data).cpu().sum()
            accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(te.dataset), 100. * accuracy.avg))

    return accuracy.avg


def main():

    best_model = None
    best_accuray = 0.0

    for epoch in range(1, 30 + 1):

        train(epoch)
        scheduler.step()
        val_accuracy = test()

        if best_accuray < val_accuracy:
            best_model   = copy.deepcopy(model)
            best_accuray = val_accuracy

    print ("The best model has an accuracy of " + str(best_accuray))
    torch.save(best_model.state_dict(), 'salientNet.model')

if __name__ == '__main__':
    main()