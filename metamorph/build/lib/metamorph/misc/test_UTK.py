import argparse
from os.path import dirname, join

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import metamorph
from metamorph.misc.processUTK import UTKDataset
from metamorph.misc.multiNN import TridentNN
from metamorph.misc.multiNN_vgg13 import TridentNN_vgg13

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', default=512, type=int, help='(int) number of batch size for training')
parser.add_argument('-n', '--num_epochs', default=61, type=int, help='(int) number of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='(float) learning rate of optimizer')
parser.add_argument('-pt', '--pre_trained', default=False, type=bool, help='(bool) whether or not to load the pre-trained model')


'''
    Function to read in the data
    Inputs: None
    Outputs:
     - train_loader : Custom PyTorch DataLoader for training data from UTK Face Dataset
     - test_loader : Custom PyTorch DataLoader for testing UTK Face Dataset
     - class_nums : Dictionary that stores the number of unique variables for each class (used in NN)
'''
def read_data(batch_size=256):
    # Read in the dataframe
    # parent_path = dirname(dirname(__file__))
    # df_path = join(parent_path, 'data/age_gender.gz')
    df_path = '../data/age_gender.gz'
    dataFrame = pd.read_csv(df_path, compression='gzip')

    # Construct age bins
    age_bins = [0,10,15,20,25,30,40,50,60,120]
    age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)

    # Split into training and testing
    train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2, random_state=10)

    # get the number of unique classes for each group
    class_nums = {'age_num':len(dataFrame['bins'].unique()), 'eth_num':len(dataFrame['ethnicity'].unique()),
                  'gen_num':len(dataFrame['gender'].unique())}

    # Define train and test transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    # Construct the custom pytorch datasets
    train_set = UTKDataset(train_dataFrame, transform=train_transform)
    test_set = UTKDataset(test_dataFrame, transform=test_transform)

    # Load the datasets into dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Sanity Check
    for X, y in train_loader:
        print(f'Shape of training X: {X.shape}')
        print(f'Shape of y: {y.shape}')
        break

    return train_loader, test_loader, class_nums


'''
   Function to train the model
   Inputs:
     - trainloader : PyTorch DataLoader for training data
     - model : NeuralNetwork model to train
     - opt : Optimizer to train with
     - num_epoch : How many epochs to train for
    Outputs: Nothing
'''
def train(trainloader, model, opt, num_epoch, device=None):
    # Configure device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Define loss functions
    age_loss = nn.CrossEntropyLoss()
    gen_loss = nn.CrossEntropyLoss()
    eth_loss = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_epoch):
        # Construct tqdm loop to keep track of training
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        gen_correct, eth_correct, age_correct, total = 0,0,0,0    # capital l on age to not get confused with loss function
        # Loop through dataLoader
        for _, (X,y) in loop:
            # Unpack y to get true age, eth, and gen values
            # Have to do some special changes to age label to make it compatible with NN output and Loss function
            age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
            X = X.to(device)

            pred = model(X)          # Forward pass
            loss = age_loss(pred[0],age) + gen_loss(pred[1],gen) + eth_loss(pred[2],eth)   # Loss calculation

            # Backpropagation
            opt.zero_grad()          # Zero the gradient
            loss.backward()          # Calculate updates
            
            # Gradient Descent
            opt.step()               # Apply updates

            # Update num correct and total
            age_correct += (pred[0].argmax(1) == age).type(torch.float).sum().item()
            gen_correct += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_correct += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

            total += len(y)

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss = loss.item())

    # Update epoch accuracy
    gen_acc, eth_acc, age_acc = gen_correct/total, eth_correct/total, age_correct/total

    # print out accuracy and loss for epoch
    print(f'Epoch : {epoch+1}/{num_epoch},    Age Accuracy : {age_acc*100},    Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n')


'''
    Function to test the trained model
    Inputs:
      - testloader : PyTorch DataLoader containing the test dataset
      - modle : Trained NeuralNetwork
    
    Outputs:
      - Prints out test accuracy for gender and ethnicity and loss for age
'''
def test(testloader, model, device=None):
    # Configure device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    size = len(testloader.dataset)
    # put the moel in evaluation mode so we aren't storing anything in the graph
    model.eval()

    age_acc, gen_acc, eth_acc = 0, 0, 0  # capital L on age to not get confused with loss function

    with torch.no_grad():
        for X, y in testloader:
            age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
            X = X.to(device)
            pred = model(X)

            age_acc += (pred[0].argmax(1) == age).type(torch.float).sum().item()
            gen_acc += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_acc += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

    age_acc /= size
    gen_acc /= size
    eth_acc /= size

    print(f"Age Accuracy : {age_acc*100}%,     Gender Accuracy : {gen_acc*100}%,    Ethnicity Accuracy : {eth_acc*100}%\n")
    return torch.tensor([age_acc, gen_acc, eth_acc])

def test_age_gen(testloader, model, device=None):
    return test_2task(testloader, model, ['age', 'gen'], device)

def test_age_eth(testloader, model, device=None):
    return test_2task(testloader, model, ['age', 'eth'], device)

def test_gen_eth(testloader, model, device=None):
    return test_2task(testloader, model, ['gen', 'eth'], device)

def test_2task(testloader, model, task=['age','gen'], device=None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    size = len(testloader.dataset)
    model.eval()

    if 'age' in task:
        task_idx_1, task1 = 0, 'Age'
        task_idx_2 = 1 if 'gen' in task else 2
        task2 = 'Gender' if 'gen' in task else 'Ethnicity'
    elif 'gen' in task:
        task_idx_1, task1 = 1, 'Gender'
        task_idx_2, task2 = 2, 'Ethnicity'
    task_acc_1, task_acc_2 = 0, 0

    with torch.no_grad():
        for X, y in testloader:
            task_y_1, task_y_2 = y[:,task_idx_1].to(device), y[:,task_idx_2].to(device)
            X = X.to(device)
            pred = model(X)

            task_acc_1 += (pred[0].argmax(1) == task_y_1).type(torch.float).sum().item()
            task_acc_2 += (pred[1].argmax(1) == task_y_2).type(torch.float).sum().item()

    task_acc_1 /= size
    task_acc_2 /= size

    print(f"{task1} Accuracy : {task_acc_1*100}%,     {task2} Accuracy : {task_acc_2*100}%\n")
    return torch.tensor([task_acc_1, task_acc_2])


'''
    Main function that stiches everything together
'''
def main():
    # Configure the device 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Read in the data and store in train and test dataloaders
    train_loader, test_loader, class_nums = read_data(args.batch_size)

    # Load the model and optimizer
    # tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
    tridentNN = TridentNN_vgg13(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])

    # Define optimizer
    opt = torch.optim.Adam(tridentNN.parameters(), lr=args.learning_rate)

    # If we are training from scratch
    if args.pre_trained == False:
        # Train the model
        train(train_loader, tridentNN, opt, args.num_epochs, device)
        # torch.save(tridentNN.state_dict(), 'model/traidentNN_epoch'+str(args.num_epochs)+'.pt')
        torch.save(tridentNN.state_dict(), '../model/traidentNN_vgg13_epoch'+str(args.num_epochs)+'.pt')
        print('Finished training, running the testing script...\n \n')
        test(test_loader, tridentNN, device)
    else:
        # Load and test the pre-trained model
        # network = torch.load('model/traidentNN_epoch'+str(args.num_epochs)+'.pt')
        network = torch.load('../model/traidentNN_vgg13_epoch'+str(args.num_epochs)+'.pt')
        tridentNN.load_state_dict(network)
        test(test_loader, tridentNN, device)


if __name__ == '__main__':
    main()