import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

from dataloader import load_data

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def test(model, val_loader):
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    model.eval()

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(val_loader):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    print("Top 1 err: ", 1 - correct_1 / len(val_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(val_loader.dataset))

    return correct_1 / len(val_loader.dataset), correct_5 / len(val_loader.dataset)


if __name__ == '__main__':
    te = load_data()

    model = models.__dict__['resnet18'](num_classes=365)
    model.load_state_dict(torch.load('sceneNet.model', map_location=device))
    model = model.to(device)
    print(model)

    acc_top1, acc_top5 = test(model, te)
    print(acc_top1, acc_top5)