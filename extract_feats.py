import sys
import cv2
import imageio
# imageio.plugins.ffmpeg.download()
import pylab
import numpy as np

sys.path.insert(0, '/usr/local/bin/caffe/python')
import skimage.transform
import os
import glob
import tqdm
import torch
from PIL import Image
import tensorflow as tf

frameNum = 40


def extract_feats():
    """Function to extract features for frames in a video.
       Input:
            filenames:  List of filenames of videos to be processes
            batch_size: Batch size for feature extraction
       Writes features in .npy files"""
    files = glob.glob("dataset/MSR-VTT/TrainValVideo/*")

    net = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')

    image_process = tf.keras.applications.inception_resnet_v2.preprocess_input

    for file in tqdm.tqdm(files):
        vid = imageio.get_reader(file, 'ffmpeg')
        curr_frames = []
        for frame in vid:
            if len(frame.shape) < 3:
                frame = np.repeat(frame, 3)
            curr_frames.append(frame)
        curr_frames = np.array(curr_frames)
        idx = np.linspace(0, len(curr_frames) - 1, frameNum).astype(int)  # get 80 frames per vid
        curr_frames = curr_frames[idx, :, :, :]
        curr_feats = []
        for frames in curr_frames:
            image_copy = np.copy(frames)
            shape = image_copy.shape
            h, w = shape[0], shape[1]
            if h > w:
                h_start = (h - w) // 2
                image_copy = image_copy[h_start:h_start + w, :]
            else:
                w_start = (w - h) // 2
                image_copy = image_copy[:, w_start:w_start + h]
            image_resized = cv2.resize(image_copy, (299, 299), interpolation=cv2.INTER_CUBIC)
            processed_image = image_process(image_resized).reshape((1, 299, 299, -1))

            output_features = net(processed_image, training=False)

            output_features = tf.transpose(output_features, perm=[0, 3, 1, 2])

            curr_feats.extend(output_features.numpy())

        curr_feats = np.array(curr_feats)
        np.save(file.replace('MSR-VTT/TrainValVideo', 'linear_feats').replace('.avi', '.npy'), curr_feats)


if __name__ == "__main__":
    extract_feats()