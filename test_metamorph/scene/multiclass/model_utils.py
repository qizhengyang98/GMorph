import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class AdaptiveConcatPool(nn.Module):
    def __init__(self, sz=(1,1)):
        super(AdaptiveConcatPool, self).__init__()
        self.sz = sz 
        self.amp = nn.AdaptiveMaxPool2d(sz)
        self.aap = nn.AdaptiveAvgPool2d(sz)
        
    def forward(self, x):
        return torch.cat((self.amp(x), self.aap(x)), dim=1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self,batch):
        return batch.view([batch.shape[0], -1])


multi_classifier_head = nn.Sequential(
                            AdaptiveConcatPool(),
                            Flatten(),
                            nn.BatchNorm1d(1024),
                            nn.Dropout(0.25),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.5),
                            nn.Linear(512, 20),
                            # nn.Sigmoid()
                        )


def get_trn_val_idxs(len_ds, val_percent=0.2):
    np.random.seed(1)
    idxs = np.random.permutation(len_ds)

    num_val_ex = int(len_ds*val_percent)
    train_idxs = idxs[:-num_val_ex]
    val_idxs = idxs[-num_val_ex:]
    return (train_idxs, val_idxs)

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True        
        
def get_resnet34_model_with_custom_head(custom_head=multi_classifier_head):
    model = resnet34(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])
    
    freeze_all_layers(model)
    
    model.add_module('custom head', custom_head)
    # model = model.to(device)
    return model

def get_model_predictions_on_a_sample_batch(model, dl):
    model.eval()
    with torch.no_grad():
        batch, actual_labels = iter(dl).next()
        batch = batch.to(device)
        actual_labels = actual_labels.to(device)
        predictions = model(batch)
    
    return (predictions, batch, actual_labels)


# #bbox utils
# def xywh_to_yxyx(ann):
#     return [ann[1], ann[0], ann[1]+ann[3], ann[0]+ann[2]]

# def yxyx_to_xywh(ann):
#     return [ann[1], ann[0], ann[3]-ann[1], ann[2]-ann[0]]


#largest item classifier + bbox utils
def combined_loss(preds, actual_labels, sz):
    pred_bboxes, pred_class_scores = preds[:,:4], preds[:, 4:]
    actual_bboxes, actual_cat_ids = actual_labels
    
    pred_bboxes = torch.sigmoid(pred_bboxes)*sz

    return F.l1_loss(pred_bboxes, actual_bboxes) + 25*F.cross_entropy(pred_class_scores, actual_cat_ids)

def get_detection_cross_entropy_loss(pred_class_scores, actual_cat_ids):
    return F.cross_entropy(pred_class_scores, actual_cat_ids).item()

def get_detection_l1_loss(pred_bboxes, actual_bboxes, sz):
    return F.l1_loss(torch.sigmoid(pred_bboxes)*sz, actual_bboxes).item()

def get_detection_accuracy(pred_class_scores, actual_cat_ids):
    pred_labels = pred_class_scores.argmax(dim=1)
    return (pred_labels==actual_cat_ids).float().mean()

def get_concat_model_summary_on_sample_set(model, dl, sz):
    correct = 0
    total_items = 0
    loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch, labels in dl:
            total_items += len(batch)
            preds = model(batch)
            loss += combined_loss(preds, labels, sz)
        
            pred_labels = preds[:,4:].argmax(dim=1)
            correct += (pred_labels==labels[1]).sum()
            
        return (correct.float()/total_items, loss.item())        


#multi class model utils
def get_multi_class_batch_accuracy(label_logits, pred_label_logits):
    return torch.all(label_logits == pred_label_logits, dim=1).float().mean()

def get_multi_class_model_summary_on_sample_set(model, dl, pred_threshold):
    correct = 0
    total_items = 0
    loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch, label_logits in dl:
            batch = batch.to(device)
            label_logits = label_logits.to(device)
            
            total_items += len(batch)
            
            pred_scores = model(batch)
            preds_probs = torch.sigmoid(pred_scores)
            # loss += multi_class_classification_loss(preds_probs, label_logits)
            
            pred_label_logits = (preds_probs >= pred_threshold).float()
            
            correct += torch.all(label_logits == pred_label_logits, dim=1).sum()
            
        return correct.float()/total_items    

def get_class_wise_metrics(model, dl, pred_threshold):
    total_predictions = torch.zeros((20))
    actual_instances = torch.zeros((20))
    correct_predictions = torch.zeros((20))
    
    model.eval()
    with torch.no_grad():    
        for batch, label_logits in dl:
            preds = model(batch)
            preds_sig = torch.sigmoid(preds)

            pred_label_logits = (preds_sig >= pred_threshold).float()

            total_predictions += pred_label_logits.sum(dim=0)
            actual_instances += label_logits.sum(dim=0)
            correct_predictions += (pred_label_logits * label_logits).sum(dim=0)
        
    return (total_predictions.numpy(), actual_instances.numpy(), correct_predictions.numpy())

def get_concat_pred_scores_and_label_logits(model, dl):
    ds_pred_scores = torch.tensor([])
    ds_gt_label_logits = torch.tensor([])
    
    model.eval()
    with torch.no_grad():        
        for batch, label_logits in dl:
            preds = model(batch)
            preds_sig = torch.sigmoid(preds)

            ds_pred_scores = torch.cat((ds_pred_scores, preds_sig))
            ds_gt_label_logits = torch.cat((ds_gt_label_logits, label_logits))
    
    return ds_pred_scores, ds_gt_label_logits

        
def print_training_loss_summary(loss, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    #prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps%n_batches)
    
    if(steps_this_epoch==1 or steps_this_epoch%print_every==0):
        print ('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}' 
               .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss))

#Test model on single image
def test_on_single_image(test_im_tensor, model, sz):
    model.eval()
    with torch.no_grad():
        preds = model(test_im_tensor)
        pred_bbox, pred_class_scores = preds[:,:4], preds[:, 4:]
        pred_bbox = torch.sigmoid(pred_bbox)*sz
        pred_cat_id = pred_class_scores.argmax(dim=1)
    return pred_bbox, pred_cat_id             