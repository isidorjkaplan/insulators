import json
from urllib import request
from PIL import Image
import numpy as np
import scipy.misc
import os
import argparse

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
import numpy as np
from PIL import Image


def main():
    if args.download:
        download_dataset(args.folder)
    train()
    pass

def train():
    #Create the generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_image_generator = train_datagen.flow_from_directory(
        args.folder + '/frames',
        batch_size=args.batch_size, target_size=(args.width, args.width),class_mode=None, seed=1, subset='training')
    train_mask_generator = train_datagen.flow_from_directory(
        args.folder + '/masks',
        batch_size=args.batch_size, target_size=(args.width, args.width),class_mode=None, seed=1, subset='training')
    #Merge to create the overall pair generator
    train_generator = zip(train_image_generator,train_mask_generator)

    # define model
    model = Unet('resnet34', encoder_weights='imagenet')
    opt = Adam(lr=args.lr)
    model.compile(opt, loss=bce_jaccard_loss, metrics=[iou_score])
    #Load the model
    if args.load:
        model.load_weights(net_file)
    #Display the model pre-training
    model.summary()
    #Train the model and save it's history
    history = model.fit_generator(train_generator,steps_per_epoch=args.steps_per_epoch, epochs=args.epoch)
    #Save the model weights
    model.save_weights(net_file)
    #Print the history as a graph
    plotHistory(history)
    if args.display:
        display(train_generator,model)


def plotHistory(history):
    # Plot history: MSE
    plt.plot(history.history['iou_score'], label='Score (testing data)')
    #plt.plot(history.history['val_iou_score'], label='Score (validation data)')
    plt.title('Accuracy on segmenting insulators')
    plt.ylabel('IOU Score')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

def display(train_generator, model):
    N = 5
    f,ax = plt.subplots(N, 3)
    for i in range(0, N):
        (img, mask) = train_generator.__next__()
        out = model.predict(img)
        out = out[0].reshape((img_width, img_height))

        ax[i,0].imshow(img[0])
        ax[i,1].imshow(out)
        ax[i,2].imshow(mask[0])
    plt.show()

def download_dataset(folder):
    #Clear the old files
    os.system("find " + folder + " -name '*.jpg' -delete")
    os.system("find " + folder + " -name '*.png' -delete")
    print("Deleted old files")
    #Some images have multiple insulators, we need to give them names, we use letters of the alphabet 
    masks = ['a','b','c', 'd', 'e', 'f']
    with open('data/data.json') as f:
        data = json.load(f)
    #For each data entry in the dataset
    for i in range(len(data)):
        #Ensure that the entry in the json file actually has an image associated with it (sometimes it does not for a stupid reason)
        if len(data[i]['Label']) == 0:
            continue;
        #Obtain the image URL
        image = data[i]['Labeled Data']
        #Retrive the image to it's proper folder location
        request.urlretrieve(image, folder + '/frames/files/image' + str(i) + '.jpg')
        #No idea what is happening here and don't care, wrote this a year or two ago
        arr = 0
        numOfObjects = len(data[i]['Label']['objects'])
        for j in range(0,numOfObjects):
            mask = data[i]['Label']['objects'][j]['instanceURI']
            mask_file = folder + '/tmp_masks/image' + str(i) + masks[j] + '.png'
            request.urlretrieve(mask, mask_file)
            img = Image.open(mask_file)
            if j == 0:
                arr = np.asarray(img)
            else:
                arr = arr + np.asarray(img)
        img = Image.fromarray(arr)
        img.save(folder + '/masks/files/image' + str(i) + '.png')
        #Increment i 
        i = i+1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', default=False, action='store_true', help='Enter to re-download the dataset')
    parser.add_argument('--folder', default='data/heatmap', help='Enter the folder for dataset training')
    #Neural network arguments
    parser.add_argument('--epoch', default=300, help='The number of epoches to run')
    parser.add_argument('--steps_per_epoch', default=5, help='The number of training entries to be included in each epoch (note we artificially extend our dataset)')
    parser.add_argument('--batch_size', default=3, help='Not really sure what this does')
    parser.add_argument('--net_file', default='data/network.h5', help='What is the file to store the resulting neural network')
    parser.add_argument('--lr', default=0.001, help='The learning rate for the neural network')
    parser.add_argument('--width', default=32*4, help='The width and height of the images for processing')
    parser.add_argument('--load', default=False, action='store_true', help='Set flag to enable loading an existing neural network')
    #Misc args
    parser.add_argument('--display', default=False, action='store_true', help='True to display the values')


    #Parse the args
    args = parser.parse_args()
    main()