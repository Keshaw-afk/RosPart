import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import message_filters
import math


def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to yaw angle."""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class StateAggregatorNode(Node):
    def __init__(self):
        super().__init__('rl_state_aggregator')

        # Cache for the dynamic goal point
        self.goal_x = None
        self.goal_y = None

        # Subscribe to the camera's goal point
        self.goal_sub = self.create_subscription(
            Point,
            '/arena/goal_point',
            self.goal_callback,
            10
        )

        # Dictionaries to hold our synchronizers and publishers
        self.syncs = {}
        self.state_pubs = {}

        for i in range(1, 5):
            robot_name = f'robot_{i}'

            # 1. Create message_filters Subscribers instead of standard rclpy Subscribers
            # Note: message_filters.Subscriber does not take a callback yet!
            pose_sub = message_filters.Subscriber(
                self, PoseStamped, f'/arena/{robot_name}/pose')
            scan_sub = message_filters.Subscriber(
                self, LaserScan, f'/arena/{robot_name}/scan_compressed')

            # 2. Create the ApproximateTimeSynchronizer
            # queue_size=10, slop=0.1 means it will match messages that arrived within 0.1 seconds of each other
            ats = message_filters.ApproximateTimeSynchronizer(
                [pose_sub, scan_sub], queue_size=10, slop=0.1)

            # 3. Register the callback using a lambda to pass the robot_name
            ats.registerCallback(lambda pose_msg, scan_msg, name=robot_name: self.sync_callback(
                pose_msg, scan_msg, name))

            # Store the synchronizer so it doesn't get garbage collected
            self.syncs[robot_name] = ats

            # Publisher for the synchronized output
            self.state_pubs[robot_name] = self.create_publisher(
                Float32MultiArray,
                f'/arena/{robot_name}/rl_state',
                10
            )

        self.get_logger().info("Message Filters State Aggregator running.")

    def goal_callback(self, msg):
        self.goal_x = msg.x
        self.goal_y = msg.y

    def sync_callback(self, pose_msg, scan_msg, robot_name):
        """This callback ONLY fires when a Pose and Scan arrive with closely matching timestamps."""

        # Don't publish anything if the camera hasn't found the goal yet
        if self.goal_x is None:
            return

        # 1. Extract Robot Pose
        rx = pose_msg.pose.position.x
        ry = pose_msg.pose.position.y
        qx = pose_msg.pose.orientation.x
        qy = pose_msg.pose.orientation.y
        qz = pose_msg.pose.orientation.z
        qw = pose_msg.pose.orientation.w

        yaw = euler_from_quaternion(qx, qy, qz, qw)

        # 2. Calculate Goal Metrics
        dx = self.goal_x - rx
        dy = self.goal_y - ry
        distance_to_goal = math.sqrt(dx**2 + dy**2)

        global_angle_to_goal = math.atan2(dy, dx)
        relative_angle = global_angle_to_goal - yaw

        # Normalize angle between -Pi and Pi
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        # 3. Assemble the Flat State Vector
        state_vector = [float(distance_to_goal), float(relative_angle)]
        state_vector.extend(scan_msg.ranges)

        # 4. Publish
        out_msg = Float32MultiArray()
        out_msg.data = state_vector
        self.state_pubs[robot_name].publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateAggregatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
