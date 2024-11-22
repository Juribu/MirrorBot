#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np

fx, fy = 640, 640 # in pixels
cx, cy = 640, 400 # in pixels


class RealSenseSubscriber:
    def __init__(self):
        rospy.init_node('realsense_subscriber', anonymous=True)

        # Topics for point cloud and RGB images
        self.point_cloud_topic = "/camera/depth/color/points"  # Adjust this based on your RealSense setup
        self.rgb_image_topic = "/camera/color/image_raw"       # Adjust this based on your RealSense setup

        # Subscribers
        self.point_cloud_sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.point_cloud_callback)
        self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self.rgb_image_callback)
        self.points_list = []
        # CVBridge to convert ROS Image messages to OpenCV format
        self.bridge = CvBridge()

    def point_cloud_callback(self, msg):
        """Callback function to process incoming PointCloud2 messages."""
        rospy.loginfo("Received a PointCloud2 message")
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        self.points_list = points_list
        # Optional: Process or save the point cloud
        rospy.loginfo("The first 10 points in our point cloud")
        for i in range(0,100000,10000):
            current_point = points_list[i]
            rospy.loginfo("Point cloud " + str(i) + " has depth " + str(current_point[2]))
        rospy.loginfo("Point cloud has %d points", len(points_list))
        

    def rgb_image_callback(self, msg):
        """Callback function to process incoming RGB image messages."""
        rospy.loginfo("Received an RGB image")
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display the image (optional)
           
            if (len(self.points_list) != 0):
                points_list = self.points_list
                rgb_made_from_point_cloud = self.pointcloud_to_image(points_list,fx,fy,cx,cy)
                #rospy.loginfo("RGB Image made from Point cloud has %d points", len(rgb_made_from_point_cloud))
                rospy.loginfo("First pixel is " + str(rgb_made_from_point_cloud[0]))
                depth_list = []
                for point in points_list:
                    depth_list.append(point[2])
                modified_image = self.draw_gradient_pixels_on_ros_image(cv_image,rgb_made_from_point_cloud, depth_list)
                cv2.imshow("Modified RGB Image", modified_image)
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)  # Add delay to update OpenCV window

        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))
    
    def pointcloud_to_image(self, points, fx, fy, cx, cy):
        """
        Converts a list of 3D points (x, y, z) from a point cloud to 2D RGB image pixels.

        Args:
            points (list of tuples): List of (x, y, z) points from the point cloud.
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Principal point in x direction.
            cy (float): Principal point in y direction.

        Returns:
            list of tuples: List of (u, v) pixel coordinates in the RGB image.
        """
        pixels = []
        for x, y, z in points:
            if z > 0:  # Ignore points behind the camera
                u = int((x / z) * fx + cx)
                v = int((y / z) * fy + cy)
                pixels.append((u, v))
        return pixels


    # Allows you to draw colored pixels (based on pixel depth) on an image (useful for knowing what pixels of the rgb camera I obtained from the PointCloud)
    def draw_gradient_pixels_on_ros_image(self, cv_image, pixel_list, depth_list):
        """
        Draws pixels with a gradient color based on depth (ROYGBIV) on an image received from a ROS topic.

        Args:
            cv_image (numpy.ndarray): The image converted from ROS Image message.
            pixel_list (list of tuples): List of (u, v) pixel coordinates.
            depth_list (list of floats): Corresponding depth values for each pixel.

        Returns:
            numpy.ndarray: The modified image with a depth-based gradient applied to pixels.
        """
        # Ensure the depth_list matches the pixel_list
        if len(pixel_list) != len(depth_list):
            raise ValueError("Pixel list and depth list must have the same length.")

        # ROYGBIV color mapping (BGR format) for gradient interpolation
        colors = [
            (0, 0, 255),    # Red
            (0, 127, 255),  # Orange
            (0, 255, 255),  # Yellow
            (0, 255, 0),    # Green
            (255, 255, 0),  # Blue
            (255, 0, 255),  # Indigo
            (128, 0, 128),  # Violet
        ]

        # Create a copy of the original image
        cv_image_copy = cv_image.copy()

        # Normalize depth values to range [0, 1] for gradient computation
        min_depth, max_depth = min(depth_list), max(depth_list)
        depth_range = max_depth - min_depth if max_depth > min_depth else 1

        for (u, v), depth in zip(pixel_list, depth_list):
            # Ensure the pixel is within the image bounds
            if 0 <= v < cv_image_copy.shape[0] and 0 <= u < cv_image_copy.shape[1]:
                # Normalize depth to [0, 1]
                normalized_depth = (depth - min_depth) / depth_range

                # Interpolate color based on normalized depth
                color_index = normalized_depth * (len(colors) - 1)
                low_idx = int(color_index)  # Lower bound index
                high_idx = min(low_idx + 1, len(colors) - 1)  # Upper bound index
                alpha = color_index - low_idx  # Interpolation factor

                # Interpolate between the two nearest colors
                color_low = np.array(colors[low_idx], dtype=float)
                color_high = np.array(colors[high_idx], dtype=float)
                interpolated_color = (1 - alpha) * color_low + alpha * color_high

                # Apply the color to the pixel
                cv_image_copy[v, u] = interpolated_color.astype(np.uint8)

        return cv_image_copy





    def run(self):
        """Keep the node running."""
        rospy.spin()

if __name__ == '__main__':
    try:
        realsense_subscriber = RealSenseSubscriber()
        realsense_subscriber.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
