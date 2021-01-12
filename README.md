# Insulators
This is a cleaned up version of the work I did in First Year with some expansions to the features

It consists of two major programs, one is `training.py` which trains the segmentation network and `evaluate.py` which uses the segmentation network to zoom in on the insulators in an image and then crop it. Instructions to use are below

# Setup
## Python Enviornment Setup
The first thing you will need to do is to setup a Python enviornment with `tensorflow` and `keras`. Instructions to do this can be found on the internet. You will also need to to install `PIL`, `segmentation_models` and `matplotlib` to the same profile. Make sure you install everything to the same profile and then use that profile to run this code. Tutorials for all that can be found online

## Dataset 
Once you clone this repository you need to place the dataset in the `data` folder to use it. You can find the full dataset I used to train and evaluate here https://1drv.ms/u/s!AvqP07DxLr0lu5cm4-PrO3CwL546Pw?e=qI0Dbq 

# Training Segmentation Network
Note, in the dataset I listed above I also have my pretrained `heatmap.h5` which is a neural network I already trained. You can also train your own from scratch if you want to modify the dataset or training parameters

Just run the training program with `python3.7 training.py`
```
usage: training.py [-h] [--download] [--folder FOLDER] [--epoch EPOCH]
                   [--steps_per_epoch STEPS_PER_EPOCH]
                   [--batch_size BATCH_SIZE] [--net_file NET_FILE] [--lr LR]
                   [--width WIDTH] [--load] [--display]

optional arguments:
  -h, --help            show this help message and exit
  --download            Enter to re-download the dataset
  --folder FOLDER       Enter the folder for dataset training
  --epoch EPOCH         The number of epoches to run
  --steps_per_epoch STEPS_PER_EPOCH
                        The number of training entries to be included in each
                        epoch (note we artificially extend our dataset)
  --batch_size BATCH_SIZE
                        Not really sure what this does
  --net_file NET_FILE   What is the file to store the resulting neural network
  --lr LR               The learning rate for the neural network
  --width WIDTH         The width and height of the images for processing
  --load                Set flag to enable loading an existing neural network
  --display             True to display the values
```

# Using the model to crop photos
To crop photos run the program `python3.7 evaluate.py`. This program will take in pictures of insulators and then zoom in on the insulators and then once it has located the insulators it will crop them out individually if possible and if not it will flag that it could not identify an insulator in the photo otherwise. 
```
usage: evaluate.py [-h] [--net_file NET_FILE] [--input INPUT]
                   [--tmp_folder TMP_FOLDER] [--output OUTPUT] [--width WIDTH]
                   [--cutoff CUTOFF] [--buffer BUFFER] [--zoom_iter ZOOM_ITER]
                   [--crop_cutoff CROP_CUTOFF]
                   [--existance_cutoff EXISTANCE_CUTOFF]
                   [--indv_buffer INDV_BUFFER]

optional arguments:
  -h, --help            show this help message and exit
  --net_file NET_FILE   What is the file to store the resulting neural network
  --input INPUT         A folder with a bunch of unsorted insulators that the
                        program will run on
  --tmp_folder TMP_FOLDER
                        A folder we will use to tinker with temporary data
  --output OUTPUT       Output data folder with cropped insulators
  --width WIDTH         The width and height of the images for processing
  --cutoff CUTOFF       This is the percentage of insulator pixels on the left
                        or right that must be contained within the insulator
  --buffer BUFFER       This is a buffer surrounding the identified insulator
                        crop box as a percentage
  --zoom_iter ZOOM_ITER
                        Number of itterations of zooming on the insulator
                        before we stop zooming
  --crop_cutoff CROP_CUTOFF
                        Average pixel value in a row for cutoff when
                        individually cropping
  --existance_cutoff EXISTANCE_CUTOFF
                        Average probability-pixel value for existance of
                        insulators
  --indv_buffer INDV_BUFFER
                        Buffer for when cropping individual insulators
```