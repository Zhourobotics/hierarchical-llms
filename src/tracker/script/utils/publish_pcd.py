#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import pcl
import pcl.pcl_visualization
import pcl_ros
import sensor_msgs.point_cloud2 as pc2
import rospkg
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import struct

def load_pcd(file_path):
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud


def publish_pcd():
    rospy.init_node('pcd_publisher', anonymous=True)
    pub = rospy.Publisher('pcd', PointCloud2, queue_size=10)

    ros_path = rospkg.RosPack().get_path('tracker')
    pcd_path = ros_path + "/data/0.pcd"
    pcl_cloud = load_pcd(pcd_path)

    # Create PointCloud2 message
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'world'
    # Convert PCL cloud to ROS PointCloud2 with RGB
    # Prepare points with colors
    points = np.asarray(pcl_cloud.points)
    colors = np.asarray(pcl_cloud.colors)

    point_cloud = []
    for i in range(len(points)):
        x, y, z = points[i]
        r, g, b = (colors[i] * 255).astype(int)
        a = 255
        rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        point_cloud.append([x, y, z, rgba])

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
    ]

    cloud_msg = pc2.create_cloud(header, fields, point_cloud)

    rate = rospy.Rate(1) # 1 Hz
    while not rospy.is_shutdown():
        cloud_msg.header.stamp = rospy.Time.now()
        pub.publish(cloud_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pcd()
    except rospy.ROSInterruptException:
        pass
