# Project: RoboND Perception Project

## 1. Introduction 

The assignment for this project was to solve the perception part of Amazon's pick and place challenge. The problem was solved in for today's standards old fashioned way by using SVM classifier for object recognition. Training samples were collected from RGBD camera simulated in Gazebo simulator. In a same way the object detection has been done as well.

## 2. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

First I have removed statistical outliers from the camera capture to address for the random distortions. I have done that by using K nearest neighbour algorithm as it can be seen in lines 59-67 of the perception.py file.

Then I have down-sampled the input cloud by using Voxel Grid Downsampling to remove unnecessary information and speed up the computation (perception.py, lines 69-78). This step required some experimentation to reach the sweet spot between number of points and information preserved.

To remove all the unnecessary information like bottom of the table and bins that will contain objects after they will be collected I have filtered input cloud with pass trough filter based on **z** and **x** axes(perception.py, lines 80-102).

The last step of the Exercise 1 was to separate objects from the tabletop area. I have done that with help of RANSAC filtering (perception.py, lines 104-124). The result were 2 point clouds, one representing objects (extracted_outliers) and the other representing tabletop (extracted_inliers).

## 3. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

As the next step I have extracted individual objects from the objects point cloud. I have done that with help of Euclidean Clustering (perception.py, lines 126-139). First I have extracted only the position related data from objects point cloud by ignoring RGB colour information with help of XYZRGB_to_XYZ function. Then I have applied K-D Three

## 4. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



