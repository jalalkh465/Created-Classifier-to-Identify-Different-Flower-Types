import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import Utility
import Model


parser = argparse.ArgumentParser(description='Train the model on given data')
parser.add_argument('data_dir', type=str, action="store", default="./flowers/", help='required directory from which to train')
parser.add_argument('--save_dir', type=str, action="store", default="./checkpoint.pth", help='optional directory to save checkpoint')
parser.add_argument('--arch', type=str, action="store", default='vgg16', help='optional architecture for features')
parser.add_argument('--learning_rate', action="store", type=float, default=0.001, help='optional learning rate')
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512, help='optional number of hidden units in classifier')
parser.add_argument('--dropout', action="store", type=float, default=0.5, help='optional dropout rate for the fully connected layer')
parser.add_argument('--epochs', type=int, action="store", default=2, help='optional number of epochs in training')
parser.add_argument('--gpu', action='store', default="gpu", help='optional choice to use GPU during training')

args = parser.parse_args()
    
data = args.data_dir
path = args.save_dir
lr = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout   
       
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    trainloader, validloader, testloader, train_data = Utility.load_data(data)
    model, criterion = Model.build_model(architecture,dropout,hidden_units,lr,power)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr= 0.001)

    
# Train Model
    epochs = 2
    steps = 0
    iteration = 3

    #training
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
        
        # move images and model to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        
        # loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % iteration == 0:
            model.eval()             #validation
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    log_ps = model.forward(inputs)
                    batch_loss = criterion(log_ps, labels)
                    valid_loss += batch_loss.item()
                    
                    #Accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape).to(device)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_loss = running_loss/len(trainloader)
            valid_loss = valid_loss/len(validloader)
            accuracy = accuracy/len(validloader)
            print("Train Loss: {:.3f}".format(train_loss))
            print("Valid Loss: {:.3f}".format(valid_loss))
            print("Accuracy: {:.3f}%".format(accuracy*100))          
    
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
                    'state_dict': model.state_dict(),
                    'epochs': epochs,
                    'optim_stat_dict': optimizer.state_dict(),
                'class_to_idx': train_data.class_to_idx
                 }
    torch.save(checkpoint, 'checkpoint.pth')
    print("checkpoint is saved!, model has been trained")
    
if __name__ == "__main__":
    main()                                                      
