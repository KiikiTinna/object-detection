#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Detect yellow color
        yellow_detection = self.detect_yellow_color(image)
        white_detection = self.detect_white_color(image)

        # Display the original and yellow-detected frames
        cv2.imshow('Original', image)
        cv2.imshow('Yellow Detection', yellow_detection)
        cv2.imshow('White DEtection', white_detection)
        cv2.waitKey(1)

    def detect_yellow_color(self, image):
        # convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # define lower and upper bounds for yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([50, 255, 255])
        # create a mask for yellow color
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Bitwise-AND mask and original image
        yellow_detection = cv2.bitwise_and(image, image, mask=yellow_mask)
        return yellow_detection
    
    def detect_white_color(self, image):
        # convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
        # define lower and upper bounds for white color
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 25, 255])
        # create a mask for white color
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        white_detection = cv2.bitwise_and(image,image,mask=white_mask)
        return white_detection


if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()