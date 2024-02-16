import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from transformers import ViTForImageClassification, AdamW

import warnings
warnings.filterwarnings("ignore")


kwargs = {'num_workers': 4, 'pin_memory': True}
bs = 8
seed = 10
torch.manual_seed(seed)
NUM_CLASSES = 5

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples]).type(torch.LongTensor)
    return {"pixel_values": pixel_values, "labels": labels}

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])

transform_test  = transforms.Compose([transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

class ESOSDataset(Dataset):
    def __init__(self, loc='./dataset/ESOS/', split='train', transform=transform_train, target_transform=None):
        self.loc = loc
        mat = sio.loadmat(f'{self.loc}imgIdx.mat')['imgIdx'][0]
        X_train, Y_train = [], []
        X_test, Y_test = [], []
        for i in mat:
            if i[2] == 0:
                X_train.append(i[0][0])
                Y_train.append(i[1][0,0])
            else:
                X_test.append(i[0][0])
                Y_test.append(i[1][0,0])
        if split=='train':
            self.X = np.array(X_train)
            self.Y = np.array(Y_train)
        else:
            self.X = np.array(X_test)
            self.Y = np.array(Y_test)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.X[idx]
        image = Image.open(f'{self.loc}img/{img_name}')
        image = image.convert('RGB')
        y = self.Y[idx]
        y = torch.Tensor([float(y)])
        image_x = self.transform(image)
        if self.target_transform:
            y = self.target_transform(y)
        return [image_x, y]

def load_data(path, bs=bs, kwargs=kwargs):
    train_loader = torch.utils.data.DataLoader(ESOSDataset(path, 
                                                split='train', transform=transform_train), 
                                                batch_size=bs, shuffle=True, collate_fn=collate_fn, **kwargs)                                   
    test_loader = torch.utils.data.DataLoader(ESOSDataset(path, 
                                                split='test', transform=transform_test), 
                                                batch_size=bs, shuffle=False, collate_fn=collate_fn, **kwargs)
    return train_loader, test_loader


def target_transform_sampler(x):
    target = torch.tensor([0]).long()
    return target

def load_data_sampler(path):
    train_data = ESOSDataset(path, 
                            split='train', transform=transform_train, target_transform=target_transform_sampler)
    return train_data

class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=NUM_CLASSES):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                              num_labels=num_labels,
                                                              ignore_mismatched_sizes=True)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        label_logits = F.one_hot(labels.to(torch.int64), 5)
        pred_dist = torch.sigmoid(logits)
        mAP = average_precision_score(label_logits.cpu().detach().numpy(), pred_dist.cpu().detach().numpy(), average='micro')

        return loss, accuracy, mAP
      
    def training_step(self, batch, batch_idx):
        loss, accuracy, mAP = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        self.log("training_accuracy", accuracy, prog_bar=True)
        self.log("training_mAP", mAP, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, mAP = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("validation_mAP", mAP, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, mAP = self.common_step(batch, batch_idx)     
        self.log("Test_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("Test_mAP", mAP, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    # def train_dataloader(self):
    #     return train_dataloader

    # def val_dataloader(self):
    #     return val_dataloader

    # def test_dataloader(self):
    #     return test_dataloader

def main():
    # multi-label classification + VOC2007
    salient_train_loader, salient_test_loader = load_data("../../datasets/ESOS/")
    # salient_train_data_sampler = load_data_sampler("../../datasets/ESOS/")

    batch = next(iter(salient_train_loader))
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    assert batch['pixel_values'].shape == (bs, 3, 224, 224)
    assert batch['labels'].shape == (bs,)
    print(next(iter(salient_test_loader))['pixel_values'].shape)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    model = ViTLightningModule.load_from_checkpoint("salient/lightning_logs/version_10273120/checkpoints/epoch=3-step=5484.ckpt")
    trainer = Trainer(default_root_dir="salient", max_epochs=4, callbacks=[EarlyStopping(monitor='validation_loss')])
    # trainer.fit(model, salient_train_loader, salient_test_loader)

    trainer.test(model, salient_test_loader)

if __name__== '__main__':
    main()

