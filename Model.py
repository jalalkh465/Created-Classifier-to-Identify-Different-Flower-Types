'''
suggestions from project part 2 instructions
Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images.
'''

import argparse
import torch
import numpy as np
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import json
from PIL import Image
import Utility

#TODO: Build and train your network
def build_model(architecture='vgg16', dropout=0.1, hidden_units=4096, lr=0.001, device='gpu'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Architecture not supported')

    # Freeze parameters of pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier with ReLU activations and dropout
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # Move model to device
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    return model, criterion


# TODO: Save the checkpoint 
def save_checkpoint(path):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optim_stat_dict': optimizer.state_dict(),
              'class_to_idx': train_data.class_to_idx
             }

#To Load the saved checkpoint
def load_checkpoint(path):
    checkpoint = torch.load('checkpoint.pth')
    model = torchvision.models.vgg16(pretrained=True)
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, topk=5, device='gpu'):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #TODO: Implement the code to predict the class from an image file    
    
    #move the model to gpu
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    #set the eval mode
    model.eval()
    
    #preprocess the image and convert to numpy array 
    img = process_image(image_path).numpy()
    #numpy array converted to pytorch tensor
    img = torch.from_numpy(np.array([img])).float()
    
    #run the prediction on the input image
    with torch.no_grad():
        output = model.forward(img.cuda())
    #get top predictions 
    probability = torch.exp(output).data
    
    return probability.topk(topk)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image