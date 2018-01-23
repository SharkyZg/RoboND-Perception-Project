#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    pcl_msg = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering used to clean up input image clouds from random distortions(outliers).
    outlier_filter = pcl_msg.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(5)
    # Set threshold scale factor
    x = .5
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Calling outlier filter function
    cloud_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    # Voxel Grid filter object created for input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # By experimenting I have found out that the leaf size of 0.01 does pretty decend job in point cloud 
    # downsampling.
    LEAF_SIZE = 0.01
    # Setting the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Calling the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # PassThrough Filter
    # PassThrough filter object for the z axis. This filter will remove buttom of the table that is
    # not relevant for the object recognition
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assigned z axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # Filter function called to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    # Pass Through filter object for the x axis.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assigned x axis and range to the passthrough filter object.
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 0.8
    passthrough.set_filter_limits(axis_min, axis_max)
    # Filter function called to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

     # RANSAC plane segmentation
    # Segmentation object created
    seg = cloud_filtered.make_segmenter()
    # Setting fitting model
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Setting max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Calling the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extracting inliners (tabletop area)
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    # Converting PLC message to ROS
    ros_table_msg = pcl_to_ros(extracted_inliers)

    # Extract outliers (objects)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    # Converting PLC message to ROS
    ros_objects_msg = pcl_to_ros(extracted_outliers)

    # Euclidean Clustering (extracting separate objects)
    # Removing colour information from the object cloud
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    # Creating cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Setting tolerances for distance threshold as well as minimum and maximum cluster size one object can have
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1500)
    # Searching the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extracting indices for each of the discovered clusters(objects)
    cluster_indices = ec.Extract()

    # Creating Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    # Creating empty lists
    color_cluster_point_list = []
    detected_objects_labels = []
    detected_objects = []
    # Looping thorugh point clouds of extracted objects
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
        # Grabbing the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(indices)
        # Converting the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extracting histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Making the prediction with help of SVM classifier which object has been observed and extracting label.
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        # Adding an extracted label to the detected_objects_labels list
        detected_objects_labels.append(label)
        # Publishing a label into RViz to be shown above an observed object
        label_pos = list(white_cloud[indices[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, j))
        # Adding the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(
        len(detected_objects_labels), detected_objects_labels))

    # Creating new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Converting PCL data to ROS messages that can be published
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publishing ROS messages
    pcl_objects_pub.publish(ros_objects_msg)
    pcl_table_pub.publish(ros_table_msg)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Invoking pr2_mover() function and sending detected objects to it
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def pr2_mover(detected_objects):

    # Initializing empty list that will contain YAML dictionaries with coordinates and 
    # other properties of detected objects
    dict_list = []

    # Saving required ROS parameters to variables
    object_list_param = rospy.get_param('/object_list') # List of objects to be collected
    place_pose_list_param = rospy.get_param('/dropbox') # Locations of bins where objects should be placed
  
    # Looping through the pick list
    for i in range(len(object_list_param)):
        # Parsing parameters into individual variables
        object_label = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        object_centroid = [] # Empty list for object centroid
        
        # Getting a Point Cloud for a given object and obtaining it's centroid
        for detected_object in detected_objects:
            # In case that object from the list is matched to detected object obtaining centroid
            # of the detected object
            if detected_object.label == object_label:
                points_arr = ros_to_pcl(detected_object.cloud).to_array()
                # Calculating centroid
                numpy_points = np.mean(points_arr, axis=0)[:3] 
                for numpy_point in numpy_points:
                    # Converting centroid to float and adding to x,y,z list
                    object_centroid.append(np.asscalar(numpy_point)) 
                break
        
        # If object is matched and centroid exists preparing ROS message
        if object_centroid != []:
            # Defining object name
            object_name = String()
            object_name.data = object_label
            # Assigning the arm to be used for pick_place
            which_arm = String()
            place_pose = Pose()
            if object_group == 'green':
                which_arm.data = 'right'
                # Creating 'place_pose' for the right arm
                place_pose.position.x = place_pose_list_param[1]['position'][0]
                place_pose.position.y = place_pose_list_param[1]['position'][1]
                place_pose.position.z = place_pose_list_param[1]['position'][2]

            else:
                which_arm.data = 'left'
                # Creating 'place_pose' for the left arm
                place_pose.position.x = place_pose_list_param[0]['position'][0]
                place_pose.position.y = place_pose_list_param[0]['position'][1]
                place_pose.position.z = place_pose_list_param[0]['position'][2]

            # Adding pick_pose for the object based on centroids
            pick_pose = Pose()
            pick_pose.position.x = object_centroid[0]
            pick_pose.position.y = object_centroid[1]
            pick_pose.position.z = object_centroid[2]
            # Creating a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            yaml_dict = make_yaml_dict(
                test_scene_num, which_arm, object_name, pick_pose, place_pose)
            dict_list.append(yaml_dict)



            # # Wait for 'pick_place_routine' service to come up
            # rospy.wait_for_service('pick_place_routine')

            # try:
            #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            #     # Insert your message variables to be sent as a service request
            #     resp = pick_place_routine(test_scene_num, object_name, which_arm, pick_pose, place_pose)

            #     print ("Response: ",resp.success)

            # except rospy.ServiceException, e:
            #     print "Service call failed: %s"%e


        else:
            print("Object from the list not found on the table!")      
  
    # Outputing request parameters into output yaml file
    output_file_name = 'output_' + str(test_scene_num.data) + '.yaml'
    send_to_yaml(output_file_name, dict_list)

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Creating Subscribers
    pcl_sub = rospy.Subscriber(
        "/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Creating Publishers
    pcl_objects_pub = rospy.Publisher(
        "/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher(
        "/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher(
        "/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher(
        "/detected_objects", PointCloud2, queue_size=1)

    # Loading SVM Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initializing empty color_list
    get_color_list.color_list = []

    # Setting test scene
    test_scene_num = Int32()
    test_scene_num.data = x

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
