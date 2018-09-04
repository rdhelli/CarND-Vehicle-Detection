import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import time
import random
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def convert_color(img, cspace='RGB'):
    if cspace == 'RGB':
        return img
    if cspace == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if cspace == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if cspace == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if cspace == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if cspace == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        feats, hog_im = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                            transform_sqrt=False, visualize=vis, feature_vector=feature_vec)
        return feats, hog_im
    # Otherwise call with one output
    else:
        feats = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                    transform_sqrt=False, visualize=vis, feature_vector=feature_vec)
        return feats


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, cspace2=None):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        if cspace2 is not None:
            if cspace2 == 'RGB':
                feature_image2 = np.copy(image)
            elif cspace2 == 'HSV':
                feature_image2 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace2 == 'LUV':
                feature_image2 = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace2 == 'HLS':
                feature_image2 = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace2 == 'YUV':
                feature_image2 = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace2 == 'YCrCb':
                feature_image2 = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        feature_shapes = {}
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        feature_shapes['spat'] = spatial_features.shape
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        if cspace2 is not None:
            np.concatenate((hist_features, color_hist(feature_image2, nbins=hist_bins, bins_range=hist_range)))
        feature_shapes['chist'] = hist_features.shape
        # Call get_hog_features()
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                     cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, int(hog_channel)], orient, pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=True)
        feature_shapes['hog'] = hog_features.shape
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features, feature_shapes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions)
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def train_classifier(cars, notcars, sample_size, clf_type, spatial_size, hist_bins, cspace,
                     orient, pix_per_cell, cell_per_block, hog_channel, cspace2=None):
    '''Extract customized features from images and train a classifier on them'''
    if sample_size is not None:
        cars = random.sample(cars, sample_size)
        notcars = random.sample(notcars, sample_size)

    t = time.time()
    car_features, feat_shape = extract_features(cars, cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins,
                                                hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block, hog_channel=hog_channel, cspace2=cspace2)
    notcar_features, _ = extract_features(notcars, cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins,
                                          hist_range=(0, 256), orient=orient, pix_per_cell=pix_per_cell,
                                          cell_per_block=cell_per_block, hog_channel=hog_channel, cspace2=cspace2)
    t2 = time.time()
    time_extract = round(t2-t, 2)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    if clf_type == 'svc_c_high':
        clf = SVC(kernel='linear', C=1.0)
    if clf_type == 'svc_c_med':
        clf = SVC(kernel='linear', C=0.01)
    if clf_type == 'svc_c_low':
        clf = SVC(kernel='linear', C=0.001)
    if clf_type == 'tree':
        clf = DecisionTreeClassifier(min_samples_split=20)
    if clf_type == 'nn':
        clf = MLPClassifier(alpha=1)
    if clf_type == 'naive_bayes':
        clf = GaussianNB()
    # Check the training time for the clf
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    time_train = round(t2-t, 2)
    # Check the score of the clf
    accuracy = round(clf.score(X_test, y_test), 4)
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 100
    indices = random.sample(range(len(X_test)), n_predict)
    example_predictions = list(clf.predict([X_test[i] for i in indices]))
    example_labels = [y_test[i] for i in indices]
    t2 = time.time()
    time_predict = round(t2-t, 4)
    return clf, X_scaler, feat_shape, accuracy, time_extract, time_train, time_predict
