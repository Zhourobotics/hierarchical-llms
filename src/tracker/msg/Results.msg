# the results to contain all the information about the tracking
std_msgs/Header header
int32 id
geometry_msgs/Point[] robot_pos
geometry_msgs/Point[] robot_vel
geometry_msgs/Point[] target_pos
geometry_msgs/Point[] robot_cmd
int32 exitflag
int32[] known_typeI_flags
int32[] known_typeII_flags
int32[] attacked_typeI_flags
int32[] attacked_typeII_flags
float32 trace
float32[] weights
int32[] assignment_matrix