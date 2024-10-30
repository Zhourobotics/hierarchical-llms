
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import scipy.special
import numpy as np

class VisualizerROS:


    def __init__(self, frame_id, drone_cmd_pubs, drone_vis_odom_pubs, target_vis_odom_pubs):

        self.target_vis_odom_pubs = target_vis_odom_pubs
        self.drone_vis_odom_pubs = drone_vis_odom_pubs
        self.drone_cmd_pubs = drone_cmd_pubs
        self.frame_id = frame_id


        self.typeI_zones_pub = rospy.Publisher('typeI_zones', MarkerArray, queue_size=10)
        self.typeII_zones_pub = rospy.Publisher('typeII_zones', MarkerArray, queue_size=10)
    


    def publish_all_others(self, target_odom, typeI_zones, typeII_zones, known_typeI_flags, known_typeII_flags):
        """
        publish all data to ROS
        """
        for i in range(len(self.target_vis_odom_pubs)): 
            self.publish_odom(self.target_vis_odom_pubs[i], target_odom[i], False)

        self.publish_typeI_zones(typeI_zones, known_typeI_flags)
        self.publish_typeII_zones(typeII_zones, known_typeII_flags)

        return


    def publish_all(self, target_odom, drone_odom, drone_cmd, robotHeights,
                    typeI_zones, typeII_zones, known_typeI_flags, known_typeII_flags):
        """
        publish all data to ROS
        """
        for i in range(len(self.target_vis_odom_pubs)): 
            self.publish_odom(self.target_vis_odom_pubs[i], target_odom[i], False)

        for i in range(len(self.drone_vis_odom_pubs)):
            drome_odom_i = np.array([drone_odom[i][0], drone_odom[i][1], robotHeights[i]])
            self.publish_odom(self.drone_vis_odom_pubs[i], drome_odom_i)
        for i in range(len(self.drone_cmd_pubs)):
            self.publish_cmd(self.drone_cmd_pubs[i], drone_cmd[i])
        
        self.publish_typeI_zones(typeI_zones, known_typeI_flags)
        self.publish_typeII_zones(typeII_zones, known_typeII_flags)

        return
    

    def publish_goal(self, goal, publisher):


        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal[0]
        marker.pose.position.y = goal[1]
        marker.pose.position.z = goal[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 0.8
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = min(1, 0.1 * goal[2])

        publisher.publish(marker)

    # input is the position of the known danger
    def publish_typeI_zones(self, danger_zones, known_flag):
        """
        publish known type I danger to ROS
        """
        size = len(danger_zones["typeI_mu"])
        if size == 0:
            return
        rd = scipy.special.erfinv(1 - 2 * 0.0001) * np.sqrt(2)
        markerArray = MarkerArray()
        for i in range(size):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "danger_zones"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            #print("danger_zones: ", danger_zones["typeI_mu"])
            #print("size: ", size)
            marker.pose.position.x = danger_zones["typeI_mu"][i][0]
            marker.pose.position.y = danger_zones["typeI_mu"][i][1]
            marker.pose.position.z = -0.5
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            #print("danger_zones: ", danger_zones["typeI_cov"])
            marker.scale.x = rd * 2 * danger_zones["typeI_cov"][i][0]
            marker.scale.y = rd * 2 * danger_zones["typeI_cov"][i][1]
            marker.scale.z = 0.5
            marker.color.a = 0.8
            if known_flag[:, i].all() == 1:
                marker.color.r = 1
                marker.color.g = 0
                marker.color.b = 0
            else:
                marker.color.r = 0.52
                marker.color.g = 0.52
                marker.color.b = 0.52
            markerArray.markers.append(marker)
        self.typeI_zones_pub.publish(markerArray)

        return
    
    def publish_typeII_zones(self, danger_zones, known_flag):
        size = len(danger_zones["typeII_mu"])
        if size == 0:
            return
        rd = scipy.special.erfinv(1 - 2 * 0.0001) * np.sqrt(2)
        markerArray = MarkerArray()
        for i in range(size):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "danger zones"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = danger_zones["typeII_mu"][i][0]
            marker.pose.position.y = danger_zones["typeII_mu"][i][1]
            marker.pose.position.z = -0.8
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = rd * 2 * danger_zones["typeII_cov"][i][0]
            marker.scale.y = rd * 2 * danger_zones["typeII_cov"][i][1]
            marker.scale.z = 0.5
            marker.color.a = 0.8
            if known_flag[:, i].all() == 1:
                marker.color.r = 0
                marker.color.g = 0
                marker.color.b = 1
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
            markerArray.markers.append(marker)
        self.typeII_zones_pub.publish(markerArray)

        return

    
    def publish_odom(self, pub, odom, is_robot=True):
        """
        publish odom to ROS
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = self.frame_id
        #print("publish_odom: {}, and frame_id: {}, dim: {}, is robot: {}".format(odom, self.frame_id, len(odom), is_robot))
        dim = len(odom)

        if dim == 3:
            odom_msg.pose.pose.position.x = odom[0]
            odom_msg.pose.pose.position.y = odom[1]
            odom_msg.pose.pose.position.z = odom[2]
        elif dim == 2:
            odom_msg.pose.pose.position.x = odom[0]
            odom_msg.pose.pose.position.y = odom[1]
            odom_msg.pose.pose.position.z = is_robot == True and 1.2 or 0
        
        odom_msg.pose.pose.orientation.x = 0
        odom_msg.pose.pose.orientation.y = 0
        odom_msg.pose.pose.orientation.z = 0
        odom_msg.pose.pose.orientation.w = 1
        pub.publish(odom_msg)
        
        return
    
    def publish_cmd(self, pub, cmd):
        """
        publish cmd to ROS
        """
        cmd_msg = Twist()
        
        dim = len(cmd)
        if dim == 3:
            cmd_msg.linear.x = cmd[0]
            cmd_msg.linear.y = cmd[1]
            cmd_msg.linear.z = cmd[2]
        elif dim == 2:
            cmd_msg.linear.x = cmd[0]
            cmd_msg.linear.y = cmd[1]
            cmd_msg.linear.z = 0
        
        pub.publish(cmd_msg)
        
        return