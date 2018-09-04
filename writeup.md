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
[video1]: ./output.mp4
[video2]: ./combined_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### 1 Writeup / README

#### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### 2 Histogram of Oriented Gradients (HOG)

#### 2.1 Explain how (and identify where in your code) you extracted HOG features from the training images.

The defined functions for this step are contained in lines #15 through #122 of the file called `utils.py`, from where they are called in the main file, `project.py` in the function `find_cars()` in lines #123 through #180

The process is started by reading in all the `vehicle` and `non-vehicle` images. Here some additional images are presented (with higher resolution) to better demonstrate the pipeline. Here are two examples of what kind of images can be found in the two classes. These images are grouped into the respective lists of titles.

Car image                  |  Not car image
:-------------------------:|:-------------------------:
![image1]                  |  ![image2]

To explore the various color spaces, I used 3d point cloud generator function `plot3d()` provided in Lesson 23 / 15. Explore Color Spaces. I have tested full images with both vehicle and non-vehicle regions, and also cropped versions, to get a feel for the possibilities.

Here is an example of the different channels of the YUV color space.

Car image                  |  Not car image
:-------------------------:|:-------------------------:
![image9]                  |  ![image10]
![image13]                 |  ![image14]
![image17]                 |  ![image18]


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2.2 Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

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

