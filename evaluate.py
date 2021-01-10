from keras.optimizers import Adam
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import os
from random import random
import argparse



def main():
    #Load the model from the save file
    model = get_model()
    #Clone our database to a version we can tinker with
    clone_data()
    #Define the image loader to read in the images from our file
    for _ in range(args.zoom_iter):
        itterate(model)
    pass

def itterate(model):
    img, _, files = get_data()
    #Generate a set of heatmaps for the given data
    out = model.predict(img)
    crop_images(files, [get_bounds(out[i]) for i in range(out.size[0])])
    pass

def crop_images(files, bounds):
    for i in range(len(bounds)):
        file = files[i]
        bound = bounds[i]
        #Crop the file
        im = Image.open(file)
        width, height = im.size
        region = (bound[0]*width, bound[1]*height, bound[2]*width, bound[3]*height)
        im = im.crop((left, top, right, bottom)) 
        im.save(file)
    pass


def get_bounds(heatmap):
    left,top=0
    right,bottom=1
    #TODO actually compute the bounds. This is the complicated part
    #Note, bounds must be returned as a number from 0 to 1, it is a percentage
    return (left,top,right,bottom)

def clone_data():
    #TODO create a function that makes a copy of the dataset in the tmp directory, we will continiously be cropping this
    pass

def get_data():
    data_gen = ImageDataGenerator(rescale=1. / 255, shuffle=False)
    batch_size = len(os.listdir(args.input + "/Broken")) + len(os.listdir(args.input + "/Healthy"))
    gen = data_gen.flow_from_directory(
        args.tmp_dir,
        batch_size=batch_size, target_size=(args.width,args.width), class_mode='binary', seed=1)
    
    img, labels = image_generator.next()
    files = gen.filenames[idx : idx + gen.batch_size]
    return (img, labels, files)

def get_model():
    model = Unet(BACKBONE, encoder_weights='imagenet')
    model.load_weights(args.net_file)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_file', default='data/network.h5', help='What is the file to store the resulting neural network')
    parser.add_argument('--input', default='data/images', help='A folder with a bunch of unsorted insulators that the program will run on')
    parser.add_argument('--masks', default='data/masks', help='A folder where we will store the output masks for each input image')
    parser.add_argument('--masks', default='data/tmp', help='A folder we will use to tinker with temporary data')
    parser.add_argument('--width', default=32*4, help='The width and height of the images for processing')
    #Parse the args
    args = parser.parse_args()
    #Run the main program
    main()