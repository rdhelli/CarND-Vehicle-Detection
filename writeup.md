# Writeup


## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/0_0_car_example.jpg
[image2]: ./output_images/1_0_notcar_example.jpg
[image3]: ./output_images/test1_5_bbox.jpg
[image4]: ./output_images/test4_5_bbox.jpg
[image5]: ./output_images/0_1_spatial.jpg
[image6]: ./output_images/1_1_spatial.jpg
[image7]: ./output_images/test1_6_heat.jpg
[image8]: ./output_images/test4_6_heat.jpg
[image9]: ./output_images/0_2_ch1.jpg
[image10]: ./output_images/1_2_ch1.jpg
[image11]: ./output_images/test2_5_bbox.jpg
[image12]: ./output_images/test5_5_bbox.jpg
[image13]: ./output_images/0_2_ch2.jpg
[image14]: ./output_images/1_2_ch2.jpg
[image15]: ./output_images/test2_6_heat.jpg
[image16]: ./output_images/test5_6_heat.jpg
[image17]: ./output_images/0_2_ch3.jpg
[image18]: ./output_images/1_2_ch3.jpg
[image19]: ./output_images/test3_5_bbox.jpg
[image20]: ./output_images/test6_5_bbox.jpg
[image21]: ./output_images/0_3_hog.jpg
[image22]: ./output_images/1_3_hog.jpg
[image23]: ./output_images/test3_6_heat.jpg
[image24]: ./output_images/test6_6_heat.jpg
[image25]: ./output_images/capture.jpg
[video1]: ./output.mp4
[video2]: ./combined_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## 1 Writeup / README

### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

## 2 Histogram of Oriented Gradients (HOG)

### 2.1 Explain how (and identify where in your code) you extracted HOG features from the training images.

The defined functions for this step are contained in lines #15 through #122 of the file called `utils.py`, from where they are called in the main file, `project.py` in the function `find_cars()` in lines #123 through #180

The process is started by reading in all the `vehicle` and `non-vehicle` images. Here some additional images are presented (with higher resolution) to better demonstrate the pipeline. Here are two examples of what kind of images can be found in the two classes. These images are grouped into the respective lists of titles.

Car image                  |  Not car image
:-------------------------:|:-------------------------:
![image1]                  |  ![image2]

#### 2.1.1 Color histogram features

To explore the various color spaces, I used 3d point cloud generator function `plot3d()` provided in Lesson 23 / 15. Explore Color Spaces. I have tested full images with both vehicle and non-vehicle regions, and also cropped versions, to get a feel for the possibilities.

Here is an example of the different channels of the YUV color space. The histogram of the different channels proved useful in the classification process as input features. Many color spaces were more prone to the change in light.

Channel | Car image                  |  Not car image
:------:|:--------------------------:|:-------------------------:
 Y      | ![image9]                  |  ![image10]
 U      | ![image13]                 |  ![image14]
 V      | ![image17]                 |  ![image18]

The function can be found at line #53 in `utils.py`. The main parameters here were:

`hist_bins` - which is the number of bins used in the histogram to group the pixel intensities.

`cspace` - which is the used color space, selected from: RGB, HSV, LUV, HLS, YUV, YCrCb

#### 2.1.2 Spatial features

The raw pixel intensities in themselves can facilitate the classification process as well. In order to reduce complexity, the pixel values are spatially binned together, which is the same as decreasing their resolution. Going down to 32×32 or further, the image still contains valuable information. Here is an example of the results:

Car image                  |  Not car image
:-------------------------:|:-------------------------:
![image5]                  |  ![image6]

The function can be found at line #46 in `utils.py`. The main parameters here were:

`spatial_size` - which is the decreased resolution of the image

#### 2.1.3 HOG features

The histogram of gradients is the method used to derive a unique signature of the typical shape of the object to be recognized. Since the gradient values are grouped into cells where their histograms are calculated based on their direction and intensity, it is robust against small shifts in rotation or translation.  The cells are then further grouped into overlapping blocks of cells, to construct the features. In this project, these features are the key components to reach accuracies of >99%. Here is an example of the resulting images:

Car image                  |  Not car image
:-------------------------:|:-------------------------:
![image21]                 |  ![image22]

The function can be found at line #30 in `utils.py`, based on the respective skimage function. The main parameters here were:

`cspace` - again, which is the used color space, selected from: RGB, HSV, LUV, HLS, YUV, YCrCb

`orient` - the number of directional bins for the gradients

`pix_per_cell` - how many pixels are considered a cell

`cell_per_block` - how many cells are considered a block

`hog_channel` - which color channels shall be used to extract HOG features from, selected from: All, 1, 2 or 3

#### 2.1.4 Classifier type

As an experiment, I was curious about the performance of the other types of classifiers that we learnt about, so I have made the classifier type customizable. Besides changing the C parameter of the Support Vector Machine, I have tried out a Decision Tree, a Naive Bayes classifier and a Neural Network (or Multi-layer Perceptron) as well. My finding was that the NN type proved superior over the others in accuracy and performance.

#### 2.2 Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

![image25]

| Unnamed: 0 | clf_type    | spatial_size | hist_bins | cspace | orient | pix_per_cell | cell_per_block | hog_channel | feat_shape_spat | feat_shape_chist | feat_shape_hog | sum_feat_shape | accuracy | time_extract | time_train | time_predict | 
|------------|-------------|--------------|-----------|--------|--------|--------------|----------------|-------------|-----------------|------------------|----------------|----------------|----------|--------------|------------|--------------| 
| 0          | nn          | 16           | 32        | RGB    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.989    | 43.64        | 7.46       | 0.0015       | 
| 1          | nn          | 16           | 32        | HSV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9969   | 44.88        | 4.18       | 0.002        | 
| 2          | nn          | 16           | 32        | LUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9856   | 43.47        | 5.24       | 0.001        | 
| 3          | nn          | 16           | 32        | HLS    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9941   | 43.87        | 7.49       | 0.001        | 
| 4          | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9932   | 43.93        | 4.27       | 0.002        | 
| 5          | nn          | 16           | 32        | YCrCb  | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9958   | 42.35        | 4.76       | 0.002        | 
| 6          | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9944   | 43.91        | 7.39       | 0.002        | 
| 7          | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | 0           | 768             | 96               | 144            | 1008           | 0.9921   | 28.14        | 7.73       | 0.001        | 
| 8          | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | 1           | 768             | 96               | 144            | 1008           | 0.9834   | 26.4         | 7.96       | 0.0013       | 
| 9          | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | 2           | 768             | 96               | 144            | 1008           | 0.9856   | 24.03        | 5.17       | 0.001        | 
| 10         | nn          | 16           | 32        | YUV    | 9      | 8            | 2              | ALL         | 768             | 96               | 5292           | 6156           | 0.9935   | 104.85       | 36.59      | 0.0075       | 
| 11         | nn          | 16           | 32        | YUV    | 9      | 16           | 2              | ALL         | 768             | 96               | 972            | 1836           | 0.9896   | 98.16        | 13.02      | 0.003        | 
| 12         | nn          | 16           | 32        | YUV    | 9      | 32           | 2              | ALL         | 768             | 96               | 108            | 972            | 0.9924   | 77.39        | 7.35       | 0.001        | 
| 13         | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9941   | 55.03        | 11.73      | 0.001        | 
| 14         | nn          | 24           | 32        | YUV    | 9      | 16           | 4              | ALL         | 1728            | 96               | 432            | 2256           | 0.9941   | 48.86        | 13.53      | 0.003        | 
| 15         | nn          | 32           | 32        | YUV    | 9      | 16           | 4              | ALL         | 3072            | 96               | 432            | 3600           | 0.9885   | 135.23       | 26.69      | 0.0066       | 
| 16         | nn          | 16           | 32        | YUV    | 6      | 16           | 4              | ALL         | 768             | 96               | 288            | 1152           | 0.9904   | 40.32        | 5.96       | 0.002        | 
| 17         | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9924   | 44.5         | 6.21       | 0.001        | 
| 18         | nn          | 16           | 32        | YUV    | 12     | 16           | 4              | ALL         | 768             | 96               | 576            | 1440           | 0.9918   | 47.45        | 10.48      | 0.002        | 
| 19         | nn          | 16           | 32        | YUV    | 9      | 16           | 2              | ALL         | 768             | 96               | 972            | 1836           | 0.9944   | 52.7         | 9.16       | 0.003        | 
| 20         | nn          | 16           | 32        | YUV    | 9      | 16           | 3              | ALL         | 768             | 96               | 972            | 1836           | 0.9935   | 48.8         | 9.78       | 0.003        | 
| 21         | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9893   | 45.1         | 4.11       | 0.001        | 
| 22         | nn          | 16           | 16        | YUV    | 9      | 16           | 4              | ALL         | 768             | 48               | 432            | 1248           | 0.9924   | 43.19        | 4.08       | 0.002        | 
| 23         | nn          | 16           | 24        | YUV    | 9      | 16           | 4              | ALL         | 768             | 72               | 432            | 1272           | 0.9955   | 42.01        | 5.96       | 0.001        | 
| 24         | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9955   | 41.44        | 7.12       | 0.001        | 
| 25         | svc_c_high  | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9797   | 43.73        | 22.27      | 0.1114       | 
| 26         | svc_c_med   | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9862   | 44.64        | 23.77      | 0.1208       | 
| 27         | svc_c_low   | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9899   | 44.36        | 32.71      | 0.2005       | 
| 28         | tree        | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9372   | 40.8         | 38.93      | 0.001        | 
| 29         | nn          | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9907   | 41.29        | 6.53       | 0.002        | 
| 30         | naive_bayes | 16           | 32        | YUV    | 9      | 16           | 4              | ALL         | 768             | 96               | 432            | 1296           | 0.9262   | 41.52        | 0.33       | 0.0025       | 



#### 2.3 Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### 3 Sliding Window Search

#### 3.1 Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 3.2 Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### 4 Video Implementation

#### 4.1 Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 4.2 Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### 5 Discussion

#### 5.1 Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

