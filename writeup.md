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

#### 2.2 Explain how you settled on your final choice of HOG parameters.

First, I have made it possible, that several classifiers can be trained in one run, based on a parameter table stored as a .csv file, called `test_parameters.csv`. After I have found an acceptable default value for all of the parameters, I started using a method similar to the gradient descent method, as in, comparing the results when only changing one parameter at a time (similarly to partial derivatives).

One problem worth mentioning was the trade-off between accuracy and extraction/training/classification time needs. A main goal was to keep it computationally effective, to be able to reach real-time operation. Some key observations:

- YUV and YCrCb are both good options, with all three channels
- HLS and HSV are worth considering, especially that HSV stays reasonably good with a single channel
- increasing `pix_per_cell` to 16 greatly increases speed without losing much accuracy
- `spatial_size` is still useful down to 16×16 resolution
- decreasing C parameter of SVC-s helps, but NN type classifiers are generally better

![image25]

#### 2.3 Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

#### 2.3.1 Classifier type

As an experiment, I was curious about the performance of the other types of classifiers that we learnt about, so I have made the classifier type customizable. Besides changing the C parameter of the Support Vector Machine, I have tried out a Decision Tree, a Naive Bayes classifier and a Neural Network (or Multi-layer Perceptron) as well. My finding was that the NN type proved superior over the others in accuracy and performance. The relevant code can be found at line #215, in `utils.py`.

#### 2.3.2 Training

After the above described features have been extracted (line #222-230 in `utils.py`), training data and 0/1 labels are generated for the vehicle/non-vehicle features. The data is randomly split in 80-20% to support testing on independent data. The `StandardScaler()` from sklearn was used to normalize the training and test data. Then the built-in, generally available `fit()` method was used to train the classifier on the training data. The accuracy values are calculated from the predictions on the test data. The relevant code can be found at lines #232 through #273. Accuracy values of up to 99.5% were reached.

### 3 Sliding Window Search

#### 3.1 Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have applied the HOG sub-sampling method due to its increase in processing speed. Instead of calculating HOG for every block, it needs to be calculated only once, lines #126 through #151 in `project.py`. The spatial and color histogram features are calculated as the window is sliding over the selected region, lines #170 and #171 in `project.py`.

As for the scales, it needed some experimentation as too many scales increased the number of false positives and considerably increased processing time, but not applying enough scales resulted in the disappearing of bounding boxes over specific regions. In general, I aimed at following the rule of perspective transformation. As in, determining the necessary sizes at the closest and farthest positions and then interpolating and experimenting in between.

Multiscale values |
:----------------:|
[400, 464, 1.0]   |
[416, 480, 1.0]   |
[400, 480, 1.25]  |
[424, 504, 1.25]  |
[400, 496, 1.5]   |
[432, 528, 1.5]   |
[400, 512, 1.75]  |
[432, 544, 1.75]  |
[400, 528, 2.0]   |
[432, 560, 2.0]   |
[400, 596, 3.5]   |
[464, 660, 3.5]   |

#### 3.2 Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Apart from tuning the parameters, I have applied a heatmap and a filtering based on the principle that false positives mostly come alone, while true detection boxes overlap for the most part. I have overlayed these heatmap images on the originals as a demonstration, and used the "hot" areas to finalize the estimated bounding boxes per vehicle.

Detected bounding boxes    |  Heat-based filtering
:-------------------------:|:-------------------------:
![image3]                  | ![image7]
![image11]                 | ![image15]
![image19]                 | ![image23]
![image4]                  | ![image8]
![image12]                 | ![image16]
![image20]                 | ![image24]

---

### 4 Video Implementation

#### 4.1 Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is a [link to my video result](./output.mp4)

![video1]

I have also taken the next step and combined the vehicle detection project with the advanced line finding project to form a more comprehensive detection pipeline.

Here is a [link to the combined result](./output_combined.mp4)

![video2]

#### 4.2 Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The relevant code can be found at lines #271 through #280 in `project.py`. 

When putting together the processing of the video images, I was able to also apply a memory effect with the `Box_mem()` class, which can be found in line #252 in `project.py`. It has proved particularly useful to stabilize the detections and filter out the false positives.

---

### 5 Discussion

#### 5.1 Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues I met, was that some color spaces were more prone to false detections. When I had a working pipeline it was surprising to see such a change, even though it was not reflected in the accuracy results.

Similarly to the road curvature calculations, it would be possible to estimate the relative vehicle positions (and hence the relative speeds) and plot them on the processed video image, but there are a bit too many areas where approximations were made. Therefore, they would add up to a significant error which would make this data difficult to put into use.

Another idea is to implement a check, whether a vehicle bounding box is in the Ego vehicle's lane ahead, to provide information about the freedom and constraints of the longitudinal vehicle control.

Finally, the biggest advancement to this project could be to apply a Deep Neural Network, to recognize the features on its own. 
