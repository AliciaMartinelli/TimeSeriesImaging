import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import audioModule
import soundfile as sf

import os
import warnings
warnings.filterwarnings('ignore')
import wave

import random
import librosa
import numpy as np
import soundfile as sf
# install pydub for using HighPassFilter and play
from pydub.playback import play
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
# import simpleaudio as sa
import matplotlib.pyplot as plt
#from helper import _plot_signal_and_augmented_signal
from IPython.display import Audio
import librosa.display as dsp
# import mir_eval
import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchaudio
# from torchsummary import summary 
import os

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# from torch import nn
# from torchvision import datasets
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor
import pathlib
# import torchvision
# from tflite_model_maker import audio_classifier
# import tensorflow as ts
from tqdm.auto import tqdm

plt.rcParams["axes.labelsize"] = 'medium'
plt.rcParams["axes.titlecolor"] = 'red'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 18



import config




obj = audioModule.audioPreprocessing()


def load_data(path):
    audioFiles = sorted(list(pathlib.Path(path).rglob("*.wav")))
    
    # Extract only the class label (the last part of the path)
    classes = [str(f.parent).split(os.sep)[-1] for f in audioFiles]  # Modify this line to get only '0' or '1'
    
    return audioFiles, classes



def augment_data1(audioFileNames, classes, save_path):
    global obj
    print("before: ", len(audioFileNames))
    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        signal_org = obj.readAudio(x)
        print(signal_org.shape)
        signal = signal_org

        cc = pathlib.Path(f"{save_path}//" + "//".join(str(x).split("\\")[1:]))
        path2save = os.path.join(cc.parent, cc.stem + "_" + str(idx)+ cc.suffix)
        
        augmented_signal = obj.audioAugmentation(signal)

        pathlib.Path(cc.parent).mkdir(parents=True, exist_ok=True)

        sf.write(path2save, augmented_signal, obj.sample_rate)
        sf.write(os.path.join(pathlib.Path(path2save).parent, x.name), signal_org, config.targetSampleRate)
    print("After:",len(list(pathlib.Path(save_path).rglob("*.wav"))))

def augment_data(audioFileNames, classes, save_path):
    global obj
    dictionary = {'1' :3600,  # Exoplanet class
                  '0' :12137}  # Non-Exoplanet class

    print("before: ", len(audioFileNames))
    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        signal_org = obj.readAudio(x)
        
        # Get the class for the current file (either "0" or "1")
        class_label = classes[idx]
        
        # Create the path to save the augmented file under the correct class directory
        class_save_path = os.path.join(save_path, class_label)  # Subfolders '0' and '1'
        pathlib.Path(class_save_path).mkdir(parents=True, exist_ok=True)
        
        if classes[idx] == "0":
            augmented_signal = obj.audioAugmentation1(signal_org, [0])
            augmented_file_name = f"{x.stem}_augmented{x.suffix}"
            augmented_path = os.path.join(class_save_path, augmented_file_name)
            sf.write(augmented_path, augmented_signal, config.targetSampleRate)
        else:
            augmentation_num = 12137 // dictionary[class_label] 
            
            for i in range(augmentation_num):
                augmented_signal = obj.audioAugmentation1(signal_org, [0])  # Apply augmentation
                augmented_file_name = f"{x.stem}_{i}{x.suffix}"  # Use the original file name with an index
                augmented_path = os.path.join(class_save_path, augmented_file_name)
                sf.write(augmented_path, augmented_signal, obj.sample_rate)


    print("After:", len(list(pathlib.Path(save_path).rglob("*.wav"))))


import common
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    """ Load the data """
    X, y = load_data(common.ORGINAL_AUDIO_DATASET)
    print(f"Train: {len(X)} - {len(y)}")
    """ Data augmentation """
    augment_data(X, y, common.AUG_AUDIO_DATASET)

