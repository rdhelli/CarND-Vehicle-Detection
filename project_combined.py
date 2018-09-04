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
from utils_combined import convert_color, get_hog_features, bin_spatial
from utils_combined import color_hist, add_heat, apply_threshold, draw_boxes
from utils_combined import draw_labeled_bboxes, train_classifier
#from utils_combined import corners_unwarp, perspective_transform, grad_n_color_filter
#from utils_combined import find_lane_sliding_windows, fit_poly, find_lane_around_poly
#from utils_combined import measure_curvature, measure_offset, lanes_to_road


# Camera calibration using chessboard images
# ----------------------------------------------------------------------
# chessboard dimensions of black square intersections
nx = 9
ny = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
# arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.
# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    # If not found, display message
    else:
        pass
cal_shape = cv2.imread('./camera_cal/calibration1.jpg').shape[1::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cal_shape, None, None)


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
      hist_bins, '\ncspace:', cspace, '\ncspace2:', cspace2, '\norient:', orient, '\npix_per_cell:',
      pix_per_cell, '\ncell_per_block:', cell_per_block, '\nhog_channel:', hog_channel)

# Try to load previously saved model (True) or train a new one (False)
load_model = True
if load_model and os.path.isfile('clf.pickle'):
    with open('clf.pickle', 'rb') as handle:
        model = pickle.load(handle)
        clf = model['clf']
        X_scaler = model['X_scaler']
    print('loaded previously saved clf model from pickle file')
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


# class to contain last n batch of boxes
class Box_mem():
    def __init__(self):
        self.boxes = []  # was the line detected in the last iteration?


# ----------------------------------------------------------------------
# Define a class to receive the characteristics of each lane detection
# ----------------------------------------------------------------------
class Line():
    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = []  # x values of the last n fits of the line
        self.bestx = None  # average x values of the last n lines
        self.best_fit = [None, None, None]  # average coeffs of last n fits
        self.current_fit = []  # actual coeffs of recent fit
        self.radius = None  # radius of curvature of the line in some units
        self.offset = None  # distance of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # diff in fit coeffs
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels


img_size = cv2.imread(images[0]).shape[1::-1]
p = 300  # hyperparameter to edit (horizontal) size of region of interest
# points order: top left, top right, bottom right, bottom left
# source area = relevant part of the road (known dimensions)
src = np.float32([[578, 460], [706, 460], [1120, 720], [190, 720]])
# destination area = middle rectangle (no cropping necessary)
dst = np.float32([[p, 0], [img_size[0]-p, 0],
                  [img_size[0]-p, img_size[1]], [p, img_size[1]]])
pts1 = np.array(src, np.int32).reshape(-1, 1, 2)
pts2 = np.array(dst, np.int32).reshape(-1, 1, 2)

# store transformation matrix
M = cv2.getPerspectiveTransform(src, dst)


Box_mem = Box_mem()
Left = Line()
Right = Line()
nl = 5
n = 8
M_inv = np.linalg.inv(M)
margin = 60  # width of band around previous polynomial
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def corners_unwarp(img, nx, ny, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    p = 100
    if ret:
        src = np.float32([corners[0], corners[nx-1],
                          corners[-1], corners[-nx]])
        img_size = gray.shape[::-1]
        dst = np.float32([[p, p], [img_size[0]-p, p],
                          [img_size[0]-p, img_size[1]-p], [p, img_size[1]-p]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
        return warped, M
    else:
        return None, None


def perspective_transform(img, mtx, dist, M):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped


def grad_n_color_filter(img):
    # thresholding for H channel
    h_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 0]
    h_thr_min = 30
    h_thr_max = 105
    h_bin = np.zeros_like(h_channel)
    h_bin[(h_channel > h_thr_min) & (h_channel <= h_thr_max)] = 1
    # thresholding for S channel
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    s_thr_min = 120
    s_thr_max = 255
    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= s_thr_min) & (s_channel <= s_thr_max)] = 1
    # gradient thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    thr_min = 35
    thr_max = 100
    sxbin = np.zeros_like(scaled_sobel)
    sxbin[(scaled_sobel >= thr_min) & (scaled_sobel <= thr_max)] = 1
    # generate colored image for debugging
    color_binary = np.dstack((s_bin, sxbin, h_bin)) * 255
    # apply combined thresholds
    # H term only used to filter S term during shadowy noise, hence the '&'
    combined_binary = np.zeros_like(sxbin)
    combined_binary[(h_bin == 1) & (s_bin == 1) | (sxbin == 1)] = 1
    return combined_binary, color_binary


def find_lane_sliding_windows(binary_warped):
    # hyperparameters
    nwindows = 12  # number of sliding windows
    margin = 100  # width of windows
    minpix = 50  # minimum no. of pixels to recenter window
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)
                          ).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)
                           ).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window to mean pos
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each side
    if leftx.shape[0] == 0 or rightx.shape[0] == 0:
        return None, None, None
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        return None, None, None
    # left_fit_values.append(left_fit)
    # right_fit_values.append(right_fit)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def find_lane_around_poly(binary_warped, left_fit, right_fit):
    # hyperparameter
    margin = 60  # width of band around previous polynomial
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the area of search
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


def measure_curvature(ploty, x_values):
    # If no pixels were found return None
    y_eval = np.max(ploty)
    # Fit new polynomials to x, y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix +
                      fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


def measure_offset(img_shape, last_x):
    # compute the offset from the center
    return (last_x - img_shape[1]/2) * xm_per_pix


def lanes_to_road(img, left_fitx, right_fitx, ploty):
    pts_left = np.array([np.vstack(
            (left_fitx, ploty)).astype(np.float32).T[::-1]])
    pts_right = np.array([np.vstack(
            (right_fitx, ploty)).astype(np.float32).T])
    pts_left = cv2.perspectiveTransform(pts_left, M_inv).astype(np.int32)
    pts_right = cv2.perspectiveTransform(pts_right, M_inv).astype(np.int32)
    pts = np.hstack((pts_left, pts_right))
    lane_lines = np.zeros_like(img)
    lane_lines = cv2.fillPoly(lane_lines, [pts], (0, 100, 0))
    lane_lines = cv2.polylines(lane_lines, pts_left,
                               False, (0, 255, 255), 12)
    lane_lines = cv2.polylines(lane_lines, pts_right,
                               False, (0, 255, 255), 12)
    return lane_lines


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
    heat = apply_threshold(heat, 6)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    heatmap_small = cv2.resize(heatmap, (320, 180)).astype(np.float32)
    norm = Normalize(vmin=0, vmax=24)
    heatmap_small = np.delete(cm.hot(norm(heatmap_small))*255.0, 3, 2)

    warped = perspective_transform(img, mtx, dist, M)
    combined_binary, color_binary = grad_n_color_filter(warped)
    if Left.detected and Right.detected:
        Left.allx, Left.ally, Right.allx, Right.ally, lane_pixels = \
                find_lane_around_poly(combined_binary,
                                      Left.current_fit, Right.current_fit)
        if Left.allx.shape[0] >= 1000 and Right.allx.shape[0] >= 1000:
            Left.detected, Right.detected = True, True
        else:
            Left.detected, Right.detected = False, False
    if not Left.detected and not Right.detected:
        Left.allx, Left.ally, Right.allx, Right.ally, lane_pixels = \
                find_lane_sliding_windows(combined_binary)
        if Left.allx.shape[0] >= 1000 and Right.allx.shape[0] >= 1000:
            Left.detected, Right.detected = True, True
        else:
            Left.detected, Right.detected = False, False
    if Left.detected and Right.detected:
        Left.current_fit, Right.current_fit, ploty = \
                fit_poly(lane_pixels.shape,
                         Left.allx, Left.ally,
                         Right.allx, Right.ally)
        if len(Left.recent_xfitted) >= n and len(Right.recent_xfitted) >= n:
            Left.recent_xfitted.pop(0)
            Right.recent_xfitted.pop(0)
        Left.recent_xfitted.append(Left.current_fit)
        Right.recent_xfitted.append(Right.current_fit)
        Left.best_fit = np.average(Left.recent_xfitted, axis=0)
        Right.best_fit = np.average(Right.recent_xfitted, axis=0)
        Left.offset = measure_offset(lane_pixels.shape, Left.best_fit[-1])
        Right.offset = measure_offset(lane_pixels.shape, Right.best_fit[-1])
    else:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    if any(Left.best_fit) and any(Right.best_fit):
        Left.radius = measure_curvature(ploty, Left.best_fit)
        Right.radius = measure_curvature(ploty, Right.best_fit)
        radius = ((Left.radius + Right.radius) / 2).astype(np.int32)
        if radius > 5000:
            radius = "straight"
        offset = round(Left.offset + Right.offset, 2)
        lane_lines = lanes_to_road(img, Left.best_fit, Right.best_fit, ploty)
        message1 = "Radius of curvature: {}".format(radius)
        message2 = "Offset from center: {}".format(offset)
        out_img = cv2.addWeighted(img, 1, lane_lines, 0.6, 0)
        cv2.putText(out_img, message1, (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), thickness=2)
        cv2.putText(out_img, message2, (100, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), thickness=2)
    else:
        out_img = img
    draw_img = draw_labeled_bboxes(np.copy(out_img), labels)
    draw_img[30:30+180, 850:850+320] = heatmap_small
    return draw_img


video_titles = [("output_combined.mp4", "project_video.mp4")]

for title in video_titles[:]:
    print('processing ', title[0], '...')
    output_video = title[0]
    clip1 = VideoFileClip(title[1])  # .subclip(7, 15)
    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)
