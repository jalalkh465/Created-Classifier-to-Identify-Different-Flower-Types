import numpy as np
import argparse
import torch
import torchvision
from torchvision import transforms, models
from collections import OrderedDict
import matplotlib.pyplot as plt
import os, random
import json
from PIL import Image
import Utility
import Model

# Define argument parser
def get_input_args():
    parser = argparse.ArgumentParser(description = 'Parser for predict.py')
    parser.add_argument('input', type=str, default='./flowers/test/17/image_03906.jpg', nargs='?', action="store", help='path to input image') 
    #nargs '?' means that the input argument is used to take zero or 1 value, if the user does not provide any value the default value will be picked.
    parser.add_argument('--dir', type=str, default='./flowers/', action="store", help='path to the directory containing input image')
    parser.add_argument('checkpoint', type=str, default='./checkpoint.pth', nargs='?', action="store", help='path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, dest="top_k", action="store", help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, dest="category_names", action="store", default='cat_to_name.json', help='path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', default="gpu", action="store",dest="gpu", help='use GPU for inference')
    return parser.parse_args()

# Get command line arguments
args = get_input_args()

# Extract variables from arguments
path_image = args.input
number_of_outputs = args.top_k
json_name = args.category_names
path = args.checkpoint
device = args.gpu



def main():
    
    # Load the pre-trained machine learning model from a checkpoint file
    model = Model.load_checkpoint(path)
    # Move model to device
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)
    # Load the label names from a JSON file
    with open(args.category_names, 'r') as json_file:
        label_names = json.load(json_file)

    # Use the model to make a prediction on an input image
    probabilities, indices = Model.predict(path_image, model, number_of_outputs, device)

    # Print the top `number_of_outputs` labels with their corresponding probabilities
    for i in range(number_of_outputs):
        #label_index = indices[0][i]
        label_index = indices[0][i].cpu().numpy().item()
        label_name = label_names[str(label_index + 1)]
        probability = probabilities[0][i]
        print("{} with a probability of {}".format(label_name, probability))

    print("Prediction completed")
   
if __name__== "__main__":
    main()