import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from transformers import ViTForImageClassification, AdamW


kwargs = {'num_workers': 4, 'pin_memory': True}
bs = 8
seed = 10
torch.manual_seed(seed)

CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )
NUM_CLASSES = len(CLASSES)

def target_transform(x):
    target = torch.zeros(NUM_CLASSES)
    anno = x['annotation']['object']
    if isinstance(anno, list):
        for obj in anno:
            target[CLASSES.index(obj['name'])] = 1
    else:
        target[CLASSES.index(anno['name'])] = 1
    return target

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])

transform_test  = transforms.Compose([transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def load_data(path, bs=bs, kwargs=kwargs):
    VOC_data_train = torchvision.datasets.VOCDetection(path + "trainval/",
                                                        year='2007', image_set='trainval', download=False,
                                                        transform=transform_train, target_transform=target_transform)
    VOC_data_test = torchvision.datasets.VOCDetection(path + "test/",
                                                        year='2007', image_set='test', download=False,
                                                        transform=transform_test, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(VOC_data_train, batch_size=bs, shuffle=True, collate_fn=collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(VOC_data_test, batch_size=bs, shuffle=False, collate_fn=collate_fn, **kwargs)

    return train_loader, test_loader

def target_transform_sampler(x):
    target = torch.tensor([0]).long()
    return target

def load_data_sampler(path):
    VOC_data_train = torchvision.datasets.VOCDetection(path + "trainval/",
                                                        year='2007', image_set='trainval', download=False,
                                                        transform=transform_train, target_transform=target_transform_sampler)

    return VOC_data_train

class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=20):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224',
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
        # predictions = logits.argmax(-1)
        # correct = (predictions == labels).sum().item()
        # accuracy = correct/pixel_values.shape[0]

        predictions = torch.sigmoid(logits)
        accuracy = average_precision_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average='micro')

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        self.log("training_accuracy", accuracy, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("Test_accuracy", accuracy, on_epoch=True, prog_bar=True)
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
    multi_train_loader, multi_test_loader = load_data("../../datasets/VOCDetection/")
    # multi_train_data_sampler = load_data_sampler("../../datasets/VOCDetection/")

    batch = next(iter(multi_train_loader))
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    assert batch['pixel_values'].shape == (bs, 3, 224, 224)
    assert batch['labels'].shape == (bs, NUM_CLASSES)
    print(next(iter(multi_test_loader))['pixel_values'].shape)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    model = ViTLightningModule.load_from_checkpoint("multiclass/lightning_logs/version_10273120/checkpoints/epoch=3-step=2508.ckpt")
    trainer = Trainer(default_root_dir="multiclass", max_epochs=4, callbacks=[EarlyStopping(monitor='validation_loss')])
    # trainer.fit(model, multi_train_loader, multi_test_loader)

    trainer.test(model, multi_test_loader)

if __name__== '__main__':
    main()

