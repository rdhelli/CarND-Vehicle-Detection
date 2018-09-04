import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import glob
import os
import cv2
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from utils import convert_color, get_hog_features, bin_spatial
from utils import color_hist, add_heat, apply_threshold, draw_boxes
from utils import draw_labeled_bboxes, train_classifier


# Prepare input data
cars = []  # 8792
notcars = []  # 8968
cars.extend(glob.glob('./data/vehicles/GTI_Far/*.png'))
cars.extend(glob.glob('./data/vehicles/GTI_Left/*.png'))
cars.extend(glob.glob('./data/vehicles/GTI_MiddleClose/*.png'))
cars.extend(glob.glob('./data/vehicles/GTI_Right/*.png'))
cars.extend(glob.glob('./data/vehicles/KITTI_extracted/*.png'))
notcars.extend(glob.glob('./data/non-vehicles/Extras/*.png'))
notcars.extend(glob.glob('./data/non-vehicles/GTI/*.png'))
print('Input data directories prepared...')
sample_size = None

# Training exploration for parameter searching (True/False)
explore = False
if explore:
    test_parameters = pd.read_csv('test_parameters.csv')
    for i, r in test_parameters.iterrows():
        print('Test', i)
        print('', r.clf_type, 'classifier',
              '\n', r.cspace, 'color space',
              '\n', r.spatial_size, 'spatial binning',
              '\n', r.hist_bins, 'histogram bins',
              '\n', r.orient, 'orientations',
              '\n', r.pix_per_cell, 'pixels per cell and',
              '\n', r.cell_per_block, 'cells per block',
              '\n', r.hog_channel, 'hog channel')

        clf, X_scaler, feat_shape, accuracy, time_extract, time_train, time_predict = \
            train_classifier(cars, notcars, sample_size, r.clf_type, (r.spatial_size, r.spatial_size),
                             r.hist_bins, r.cspace, r.orient, r.pix_per_cell, r.cell_per_block, r.hog_channel)

        spat = feat_shape["spat"][0]
        chist = feat_shape["chist"][0]
        fhog = feat_shape["hog"][0]
        test_parameters.loc[i, 'feat_shape_spat'] = spat
        test_parameters.loc[i, 'feat_shape_chist'] = chist
        test_parameters.loc[i, 'feat_shape_hog'] = fhog
        test_parameters.loc[i, 'sum_feat_shape'] = spat+chist+fhog
        test_parameters.loc[i, 'accuracy'] = accuracy
        test_parameters.loc[i, 'time_extract'] = time_extract
        test_parameters.loc[i, 'time_train'] = time_train
        test_parameters.loc[i, 'time_predict'] = time_predict

        print(feat_shape, 'feature shape')
        print(accuracy, 'accuracy')
        print(time_extract, 'sec to extract features...')
        print(time_train, 'sec to train classifier...')
        print(time_predict, 'sec to make predictions...')

        model = {}
        model['clf'] = clf
        model['X_scaler'] = X_scaler
        with open('clf_{}.pickle'.format(i), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    test_parameters.to_csv('test_parameters.csv', index=False)

# Training
clf_type = 'nn'  # Can be svc_c_low/med/high, tree, nn, naive_bayes
spatial_size = (16, 16)
hist_bins = 32
cspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
cspace2 = None  # Can be one from above or None
orient = 9
pix_per_cell = 16
cell_per_block = 4
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
print('Input parameters:', '\nclf type:', clf_type, '\nspatial_size:', spatial_size, '\nhist_bins:',
      hist_bins, '\n cspace:', cspace, '\ncspace2:', cspace2, '\norient:', orient, '\npix_per_cell:',
      pix_per_cell, '\ncell_per_block:', cell_per_block, '\nhog_channel:', hog_channel)

# Try to load previously saved model (True) or train a new one (False)
load_model = True
if load_model and os.path.isfile('clf.pickle'):
    with open('clf.pickle', 'rb') as handle:
        model = pickle.load(handle)
        clf = model['clf']
        X_scaler = model['X_scaler']
    print('saved clf model loaded from pickle file')
else:
    clf, X_scaler, feat_shape, accuracy, time_extract, time_train, time_predict = \
        train_classifier(cars, notcars, sample_size, clf_type, spatial_size, hist_bins,
                         cspace, orient, pix_per_cell, cell_per_block, hog_channel, cspace2)

    print('new clf model trained with following results:')
    print(feat_shape, 'feature shape')
    print(accuracy, 'accuracy')
    print(time_extract, 'sec to extract features...')
    print(time_train, 'sec to train classifier...')
    print(time_predict, 'sec to make predictions...')
    model = {}
    model['clf'] = clf
    model['X_scaler'] = X_scaler
    with open('clf.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('clf model saved to pickle file')


# Extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, clf, X_scaler, cspace, spatial_size, hist_bins,
              orient, pix_per_cell, cell_per_block, hog_channel):
    '''Detect vehicles and return containing boxes'''
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, cspace=cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        channels = [ch1, ch2, ch3]
        hog = get_hog_features(channels[hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
    return boxes


def find_cars_multiscale(img, multiscale, clf, X_scaler, cspace, spatial_size, hist_bins,
                         orient, pix_per_cell, cell_per_block, hog_channel):
    '''Find cars with several search window sizes'''
    boxes = []
    for i in range(len(multiscale)):
        ystart = multiscale[i][0]
        ystop = multiscale[i][1]
        scale = multiscale[i][2]
        boxes.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, cspace, spatial_size,
                               hist_bins, orient, pix_per_cell, cell_per_block, hog_channel))
    return boxes


images = glob.glob('./test_images/test*.jpg')
multiscale = [[400, 464, 1.0],  # ystart, ystop, scale
              [416, 480, 1.0],
              [400, 480, 1.25],
              [424, 504, 1.25],
              [400, 496, 1.5],
              [432, 528, 1.5],
              [400, 512, 1.75],
              [432, 544, 1.75],
              [400, 528, 2.0],
              [432, 560, 2.0],
              [400, 596, 3.5],
              [464, 660, 3.5]]

for fname in images:
    print('processing ', fname, '...')
    img = mpimg.imread(fname)
    boxes = find_cars_multiscale(img, multiscale, clf, X_scaler, cspace, spatial_size,
                                 hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
    out_img = draw_boxes(img, boxes)
    plt.imsave(r"./output_images/" + fname.split('\\')[-1].split('.')[0] + "_1_bbox.jpg", out_img)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, boxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # Prepare heatmap image overlay
    heatmap_small = cv2.resize(heatmap, (320, 180)).astype(np.float32)
    norm = Normalize(vmin=0, vmax=12)
    heatmap_small = np.delete(cm.hot(norm(heatmap_small))*255.0, 3, 2)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    # Insert heatmap image overlay
    draw_img[50:50+180, 50:50+320] = heatmap_small
    plt.imsave(r"./output_images/" + fname.split('\\')[-1].split('.')[0] + "_2_heat.jpg", draw_img)


# class to contain last n batch of boxes
class Box_mem():
    def __init__(self):
        self.boxes = []  # was the line detected in the last iteration?


Box_mem = Box_mem()
n = 8


def process_image(img):
    '''Pipeline to prepare video images with vehicle detection'''
    boxes = find_cars_multiscale(img, multiscale, clf, X_scaler, cspace, spatial_size,
                                 hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
    if len(Box_mem.boxes) >= n:
        Box_mem.boxes.pop(0)
    Box_mem.boxes.append(boxes)
    box_with_mem = []
    for box in Box_mem.boxes:
        box_with_mem.extend(box)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_with_mem)
    heat = apply_threshold(heat, 5)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    heatmap_small = cv2.resize(heatmap, (320, 180)).astype(np.float32)
    norm = Normalize(vmin=0, vmax=24)
    heatmap_small = np.delete(cm.hot(norm(heatmap_small))*255.0, 3, 2)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    draw_img[50:50+180, 50:50+320] = heatmap_small
    return draw_img  # , heat


video_titles = [("output.mp4", "project_video.mp4"),
                ("output2.mp4", "project_video2.mp4"),
                ("output3.mp4", "project_video3.mp4")]

for title in video_titles[:1]:
    print('processing ', title[0], '...')
    output_video = title[0]
    clip1 = VideoFileClip(title[1])  # .subclip(7, 12)
    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)
