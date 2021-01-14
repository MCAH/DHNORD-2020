#!usr/bin/env python
# coding:utf-8
"""
Name : xgb_pred_Visualizations_mcah_error_catch.py
Author : Prasham Sheth
Contact : pds2136@columbia.edu
Description : This file runs the script to execute the XGBoost classifier on the given folder of RAW images by first converting them into their respective Discrete Fourier Transformed version followed by running XGBoost model to generate a CSV file with probabilities for each of the images. It also created Visualizations with bright spots detected and a CSV file for the same as well.

"""


import os
import re
from fnmatch import fnmatch
from tkinter import Tk, filedialog
from tqdm import *

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
toplevel_dir = os.getcwd()

root = Tk()
root.withdraw()


def create_tmp_dir(classify_dir, input_dir):
    '''
    name directory next available incremental name
    saved in the classify_halftone folder
    '''
    i = 0
    name = input_dir.split("/")[-1]+"_DFT_"

    while os.path.exists(os.path.join(classify_dir, name+"{}".format(i))):
        i += 1
    tmp_path = os.path.join(classify_dir, name+"{}".format(i))
    os.mkdir(tmp_path)
    return tmp_path


def create_tmp_dir_visualizations(classify_dir, input_dir):
    '''
    name directory next available incremental name
    saved in the classify_halftone folder
    '''
    i = 0
    name = input_dir.split("/")[-1].split("_DFT")[0]+"_Processed_Visualizations_"

    while os.path.exists(os.path.join(classify_dir, name+"{}".format(i))):
        i += 1
    tmp_path = os.path.join(classify_dir, name+"{}".format(i))
    os.mkdir(tmp_path)
    return tmp_path


def crop_img(img_data):
    '''
    crop an image to account for boundary noise
    '''
    rows, cols = img_data.shape
    boundary_x = rows//8
    boundary_y = cols//8
    img_data = img_data[boundary_x:rows-boundary_x, boundary_y:cols-boundary_y]

    rows, cols = img_data.shape

    return img_data, rows, cols


def compute_dft(filename):
    '''
    filename: filename for which we want the DFT

    Reads the image from the given filename and computes the DFT for it.

    Returns: THE DFT of the image.
    '''
    try:
        img_data = cv2.imread(filename, 0)
        img_data = cv2.resize(img_data, (3200, 1800))
        rows, cols = img_data.shape
        img_data, rows, cols = crop_img(img_data)
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        nimg = np.zeros((nrows, ncols))
        nimg[:rows, :cols] = img_data

        dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * \
            np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        magnitude_spectrum *= (255.0/magnitude_spectrum.max())
        return magnitude_spectrum
    except:
        print(filename)
        return None


def create_dfts(input_dir):
    '''
    input_dir: path to pre-cropped slides

    saves computed DFT's in the tmp_path

    returns: folder path to computed dfts
    '''

    classify_dir = os.path.dirname(os.path.realpath(__file__))
    img_list = os.listdir(input_dir)

    # filter on just images in the directory
    img_list = [filename for filename in img_list
                if filename.endswith(
                    ('.jpg', '.jpeg', '.png', '.tif', '.TIF', '.TIFF')
                )]

    tmp_path = create_tmp_dir(classify_dir, input_dir)
    for filename in tqdm(img_list):
        # print(filename)
        try:
            dft = compute_dft(os.path.join(input_dir, filename))
            rows, cols = dft.shape
            cv2.imwrite(os.path.join(
                tmp_path, filename.split('.')[0] + ".png"), dft)
        except:
            print(filename)
    return tmp_path


def check_all_images(tmp_path):
    '''
    Checks to see we have passed with images only.
    This is used as a sanity check while running the classifier to make sure
    the script doesn'trun into an erroneous situtation of using other
    than DFTs for classification.
    '''

    val = True
    for f in os.listdir(tmp_path):
        if fnmatch(f, '*.jpg') or fnmatch(f, '*.png'):
            val &= True
        else:
            val = False
    return val


def run_classifier(input_dir):
    '''
    Method for running the classifier. The input_dir is the path of the folder
    containing the DFTs.
    '''
    name = os.path.basename(input_dir)
    name = name[:name.find("_DFT")]
    file_name_csv = "results_" + name + ".csv"
    if check_all_images(input_dir) == True:
        if os.path.exists(file_name_csv):
            os.remove(file_name_csv)
    else:
        print("The folder that you selected has some files which are not images")
    DATADIR = input_dir
    IMG_SIZE = 256
    data = []
    file_name = []

    def create_data_BW():
        path = os.path.join(DATADIR)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                cropped = new_array[0:IMG_SIZE//2, 0:IMG_SIZE//2]
                data.append([new_array])
                file_name.append(img)
            except Exception as e:
                pass
    create_data_BW()
    X_BW = []
    for features in data:
        X_BW.append(features)
    X_BW = np.array(X_BW).reshape(-1, IMG_SIZE*IMG_SIZE)
    loaded_model = joblib.load("finalized_model_xgboost_08_19.joblib.dat")
    result = loaded_model.predict(X_BW)
    map_result = {0: "Halftone", 1: "Non Halftone"}
    result = [map_result[k] for k in result]
    prob = loaded_model.predict_proba(X_BW)
    adjusted_labels = [1 if i >= 0.3 else 0 for i in prob[:, 1]]
    adjusted_results = [map_result[k] for k in adjusted_labels]
    df = pd.DataFrame({"File Name": file_name,
                       "Result": result,
                       "Probability Halftone": prob[:, 0],
                       "Probability Non-Halftone": prob[:, 1],
                       "Adjusted Results": adjusted_results})
    df = df.sort_values("File Name")
    df.to_csv(file_name_csv, index=False)


def create_visualizations_detect_bright_spots(input_dir):
    name = os.path.basename(input_dir)
    name = name[:name.find("_DFT")]

    DATADIR = input_dir
    classify_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = create_tmp_dir_visualizations(classify_dir, input_dir)
    final_list = []
    for img in (os.listdir(input_dir)):
        final_list.append(os.path.join(input_dir, img))

    out_dict = {
        "Filename": [],
        "Number of Points": [],
        "X": [],
        "Y": []
    }

    for i in tqdm(final_list):
        try:
            name = os.path.basename(i)
            img = cv2.imread(i, 0)
            temp_img = cv2.resize(img, (256, 256))
            neighborhood_size = 50
            threshold = 75
            data = temp_img
            data_max = filters.maximum_filter(data, neighborhood_size)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 1
            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []
            for dy, dx in slices:
                x_center = (dx.start + dx.stop - 1)/2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1)/2
                y.append(y_center)

            # display the original input image
            (fig, ax) = plt.subplots(1, 2)
            ax[0].imshow(temp_img)
            ax[0].set_title("DFT")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].imshow(data)
            ax[1].autoscale(False)
            ax[1].plot(x, y, 'ro')
            ax[1].set_title("DFT_Processed")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            fig.suptitle(name, fontsize=12)
            # show our plots

            plt.savefig(os.path.join(output_path, name.split(".")[0]+"_processed.png"))
            plt.close()
            out_dict["Filename"].append(name)
            out_dict["Number of Points"].append(len(x))
            out_dict["X"].append(x)
            out_dict["Y"].append(y)
        except:
            print(i)
    #     print(x,y)

    output_df = pd.DataFrame(out_dict)
    output_df.to_csv(os.path.join(output_path, "Results_Analysis_Visualizations.csv"))

    df = pd.read_csv(os.path.join(output_path, "Results_Analysis_Visualizations.csv"))
    for i in df.index:
        df["X"][i] = list(map(float, (df["X"][i][2:-2].split(","))))
        df["Y"][i] = list(map(float, (df["Y"][i][2:-2].split(","))))
    distance = []
    distance_value = []
    for i in df.index:
        x_list = np.asarray(df["X"][i]) - 128.0
        y_list = np.asarray(df["Y"][i]) - 128.0
        x_list2 = x_list**2
        y_list2 = y_list**2
        dist = (x_list2 + y_list2)**0.5
        distance.append(dist)
        distance_value.append(np.sum(dist))
    df["distance"] = distance
    df["distance_value"] = distance_value
    df["avg_distance_value"] = df["distance_value"]/df["Number of Points"]
    df.to_csv(os.path.join(output_path, "Results_Analysis_Visualizations.csv"))

    return output_path


def main():
    '''
    The main method for the script. When prompted the folder containing
    RAW(TIFFs) images has to be selected.

    For the selected folder firstly we create DFTs, and then pass in the path
    of the folder that we stored DFTs into and run the classifier for that folder.

    Excel file containing the results is then stored into the same location as where this script is.
    '''
    input_dir1 = filedialog.askdirectory(parent=root, initialdir=toplevel_dir,
                                         title='Select a Directory(Folder) of Images to be classified.')

    print("Converting images to their DFTs")
    path = create_dfts(input_dir1)
    print("DFTs Created")
    print("Running the Classifier")
    run_classifier(path)
    print("File saved!")
    print("Creating Visualizations")
    create_visualizations_detect_bright_spots(path)
    print("Visualizations created!")


if __name__ == "__main__":
    main()
