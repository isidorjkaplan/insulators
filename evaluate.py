from keras.optimizers import Adam
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import shutil
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
    for i in range(args.zoom_iter):
        itterate(model)
        print(str(i) + ': ' + str(times))
    crop_individually(model)
    shutil.rmtree(args.tmp_folder)

def crop_individually(model):
    img, _, files = get_data()
    out = model.predict(img)
    for i in range(len(out)):
        im = Image.open(args.tmp_folder + '/' + files[i])
        path = args.output + '/' + files[i]
        #print(np.mean(out[i]))
        if np.mean(out[i]) > args.existance_cutoff:
            insulator_num = 1
            for top,bottom in get_individual_bounds(out[i]):
                height = bottom-top
                buffer = height*args.indv_buffer
                bound = (0,np.clip(top - buffer,0,1),
                            1,np.clip(bottom + buffer,0,1))
                #TODO, set left and right bounds as well
                width, height = im.size
                region = (int(bound[0]*width), int(bound[1]*height), int(bound[2]*width), int(bound[3]*height))
                #TODO ensure that the insulators are valid by checking the average pixel value
                im.crop(region).save(path[:-4] + str(insulator_num) + '.jpg')
        else:
            im.save(path.replace('unsorted', 'error'))
    

def get_individual_bounds(heatmap):
    probs = np.array(heatmap)
    rows = np.mean(probs,axis=1)
    arr = []
    detected = False
    row = 0
    for i in range(len(rows)):
        #TODO ensure that it is not just a fluke lucky row by looking at the avg of the past few rows
        if rows[i] > args.crop_cutoff and not detected:
            detected = True
            row = i
        if rows[i] < args.crop_cutoff and detected:
            detected = False
            arr.append((row/args.width,i/args.width))
    return arr


def itterate(model):
    img, _, files = get_data()
    #Generate a set of heatmaps for the given data
    times['heatmap'] = time.process_time()
    out = model.predict(img)
    times['heatmap'] = time.process_time() - times['heatmap']
    #for i in range(len(out)):
        #As a temporary measure
    #    mask_image = Image.fromarray((out[i].reshape((args.width, args.width)) * 255).astype(np.uint8))
    #    mask_image.save('data/tmp_out/' + files[i])

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

def find_subarray(arr):
    sum_arr = np.cumsum(arr)
    sum_arr = sum_arr / sum_arr[-1]
    for i in range(len(arr)):
        if sum_arr[i] < args.cutoff:
            left = i
        if sum_arr[i] < 1 - args.cutoff:
            right = i
    return (left,right)
            



def get_bounds(heatmap):
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
    top = np.clip(top/args.width - args.buffer, 0, 1)
    bottom = np.clip(bottom/args.width + args.buffer,0,1)
    left = np.clip(left/args.width - args.buffer,0,1)
    right = np.clip(right/args.width + args.buffer,0,1)
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
    parser.add_argument('--output', default='data/output', help='Output data folder with cropped insulators')
    parser.add_argument('--width', type=int,default=32*4, help='The width and height of the images for processing')
    parser.add_argument('--cutoff', type=float,default=0.1, help='This is the percentage of insulator pixels on the left or right that must be contained within the insulator')
    parser.add_argument('--buffer', type=float, default=0.2, help='This is a buffer surrounding the identified insulator crop box as a percentage')
    parser.add_argument('--zoom_iter', type=int,default=3, help='Number of itterations of zooming on the insulator before we stop zooming')
    parser.add_argument('--crop_cutoff',type=float, default=0.8, help='Average pixel value in a row for cutoff when individually cropping')
    parser.add_argument('--existance_cutoff', type=float,default=0.1, help='Average probability-pixel value for existance of insulators')
    parser.add_argument('--indv_buffer', type=float, default=0.4, help='Buffer for when cropping individual insulators')
    times = {'heatmap':0, 'crop':0}
    #Parse the args
    args = parser.parse_args()
    #Run the main program
    main()