import torch
from torch.autograd import Variable

from facial_age_gender.VGG_Face_torch import VGG_ageNet, VGG_genderNet
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


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


def test_facial(model, test_loader, device):
    accuracy = average_meter()

    model = model.to(device).eval()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = Variable(data,volatile=True).to(device), Variable(target,volatile=True).to(device)
            output = model(data)

            pred = output.data.max(1)[1]
            prec = pred.eq(target.data).cpu().sum()
            accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accuracy.sum), len(test_loader.dataset), 100. * accuracy.avg))

    return accuracy.avg


def test_multi_acc(test_loader_list, model, device):
    accuracy = []
    for i in range(len(test_loader_list)):
        accuracy.append(average_meter())

    model = model.to(device).eval()

    with torch.no_grad():
        for i, test_loader in enumerate(test_loader_list):
            for data, target in test_loader:

                data, target = Variable(data,volatile=True).to(device), Variable(target,volatile=True).to(device)
                output = model(data)[i]

                pred = output.data.max(1)[1]
                prec = pred.eq(target.data).cpu().sum()
                accuracy[i].update(float(prec) / data.size(0), data.size(0))

    results = []
    str = ''
    for i in range(len(accuracy)):
        results.append(accuracy[i].avg)
        str += f"net{i+1} Accuracy: {accuracy[i].avg*100}%   "
    print(str + "\n")

    return torch.tensor(results)


if __name__ == '__main__':
    # main()
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 4, 'pin_memory': True}
    transform  = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                    ])
    age_data = torchvision.datasets.ImageFolder('../../datasets/adience/age',transform=transform)
    train_indices, test_indices = train_test_split(list(range(len(age_data.targets))), test_size=0.2, stratify=age_data.targets, random_state=10)
    test_data = torch.utils.data.Subset(age_data, test_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, **kwargs)
    model = VGG_ageNet()
    model.load_state_dict(torch.load('pre_models/ageNet.model', map_location=DEVICE))
    model.to(DEVICE)
    test_facial(model, test_loader, DEVICE)


    gender_data = torchvision.datasets.ImageFolder('../../datasets/adience/gender',transform=transform)
    train_indices, test_indices = train_test_split(list(range(len(gender_data.targets))), test_size=0.2, stratify=gender_data.targets, random_state=10)
    test_data = torch.utils.data.Subset(gender_data, test_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, **kwargs)
    model = VGG_genderNet()
    model.load_state_dict(torch.load('pre_models/genderNet.model', map_location=DEVICE))
    model.to(DEVICE)
    test_facial(model, test_loader, DEVICE)