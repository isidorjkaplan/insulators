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
from distutils.dir_util import copy_tree
import time



def main():
    #Load the model from the save file
    model = get_model()
    #Clone our database to a version we can tinker with
    copy_tree(args.input, args.tmp_folder)
    #Define the image loader to read in the images from our file
    for i in range(5):#TODO set a proper number of itterations
        itterate(model)
        print(str(i) + ': ' + str(times))
    pass

def itterate(model):
    img, _, files = get_data()
    #Generate a set of heatmaps for the given data
    times['heatmap'] = time.process_time()
    out = model.predict(img)
    times['heatmap'] = time.process_time() - times['heatmap']
    for i in range(len(out)):
        #As a temporary measure
        mask_image = Image.fromarray((out[i].reshape((args.width, args.width)) * 255).astype(np.uint8))
        mask_image.save('data/tmp_out/' + files[i])

    times['crop'] = time.process_time()
    crop_images(files, [get_bounds(out[i]) for i in range(len(out))])
    times['crop'] = time.process_time() - times['crop']
    pass

def crop_images(files, bounds):
    for i in range(len(bounds)):
        path = args.tmp_folder + '/' + files[i]
        bound = bounds[i]
        #Crop the file
        im = Image.open(path)
        width, height = im.size
        region = (bound[0]*width, bound[1]*height, bound[2]*width, bound[3]*height)
        im = im.crop(region) 
        im.save(path)
    pass

def find_subarray(arr, cutoff=0.1):
    sum_arr = np.cumsum(arr)
    sum_arr = sum_arr / sum_arr[-1]
    for i in range(len(arr)):
        if sum_arr[i] < cutoff:
            left = i
        if sum_arr[i] < 1 - cutoff:
            right = i
    return (left,right)
            



def get_bounds(heatmap, buffer=0.2):
    #left=top=0.1
    #right=bottom=0.9
    #TODO actually compute the bounds. This is the complicated part
    probs = np.array(heatmap)
    cols = np.mean(probs,axis=0)#.argmax() / args.width
    rows = np.mean(probs,axis=1)#.argmax() / args.width
    #Primitive way to select the values
    left,right = find_subarray(cols)
    top,bottom = find_subarray(rows)
    #Note, bounds must be returned as a number from 0 to 1, it is a percentage
    top = np.clip(top/args.width - buffer, 0, 1)
    bottom = np.clip(bottom/args.width + buffer,0,1)
    left = np.clip(left/args.width - buffer,0,1)
    right = np.clip(right/args.width + buffer,0,1)
    #clip
    return (left,top,right,bottom)


def get_data():
    data_gen = ImageDataGenerator(rescale=1. / 255)
    batch_size = sum([len(files) for r, d, files in os.walk(args.input)])
    gen = data_gen.flow_from_directory(
        args.tmp_folder,
        batch_size=batch_size, target_size=(args.width,args.width), class_mode='binary', seed=1, shuffle=False)
    
    img, labels = gen.next()
    files = gen.filenames
    return (img, labels, files)

def get_model():
    model = Unet('resnet34', encoder_weights='imagenet')
    model.load_weights(args.net_file)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_file', default='data/heatmap.h5', help='What is the file to store the resulting neural network')
    parser.add_argument('--input', default='data/images', help='A folder with a bunch of unsorted insulators that the program will run on')
    parser.add_argument('--masks', default='data/masks', help='A folder where we will store the output masks for each input image')
    parser.add_argument('--tmp_folder', default='data/tmp', help='A folder we will use to tinker with temporary data')
    parser.add_argument('--width', default=32*4, help='The width and height of the images for processing')
    times = {'heatmap':0, 'crop':0}
    #Parse the args
    args = parser.parse_args()
    #Run the main program
    main()