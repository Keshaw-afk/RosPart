import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge
import cv2
import math
import numpy as np


class ArenaPerceptionNode(Node):
    def __init__(self):
        super().__init__('arena_perception_node')

        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/top_camera/image',
            self.image_callback,
            10
        )

        # Publishers
        self.goal_pub = self.create_publisher(Point, '/arena/goal_point', 10)

        self.robot_pubs = {}

        # Arena configuration
        self.arena_size_m = 11.48

        # ArUco / AprilTag setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create() if hasattr(
            cv2.aruco, 'DetectorParameters_create') else cv2.aruco.DetectorParameters()

        self.get_logger().info("Arena Perception Node Initialized (Using cv_bridge).")

    def image_callback(self, msg):
        try:
            # Safely use cv_bridge now that NumPy is downgraded
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        H, W = cv_image.shape[:2]

        # Calculate meters per pixel
        Sx = self.arena_size_m / W
        Sy = self.arena_size_m / H

        self.process_goal(cv_image, W, H, Sx, Sy)
        self.process_robots(cv_image, W, H, Sx, Sy)

    def pixel_to_world(self, u, v, W, H, Sx, Sy):
        """Converts pixel coordinates to metric Gazebo coordinates."""
        X = ((H / 2.0) - v) * Sy
        Y = ((W / 2.0) - u) * Sx
        return float(X), float(Y)

    def process_goal(self, cv_image, W, H, Sx, Sy):
        """Extracts the green dot and publishes its world coordinate."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                world_x, world_y = self.pixel_to_world(cx, cy, W, H, Sx, Sy)

                goal_msg = Point()
                goal_msg.x = float(world_x)
                goal_msg.y = float(world_y)
                goal_msg.z = 0.0

                self.goal_pub.publish(goal_msg)

    def process_robots(self, cv_image, W, H, Sx, Sy):
        """Detects AprilTags, calculates poses, and publishes them to ID-specific topics."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if hasattr(cv2.aruco, 'detectMarkers'):
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
        else:
            self.get_logger().error("ArUco module not found in cv2.")
            return

        if ids is not None:
            for i in range(len(ids)):
                # Extract the integer ID of the tag
                tag_id = int(ids[i][0]) + 1
                tag_corners = corners[i][0]

                cx = np.mean(tag_corners[:, 0])
                cy = np.mean(tag_corners[:, 1])

                world_x, world_y = self.pixel_to_world(cx, cy, W, H, Sx, Sy)

                top_mid_x = (tag_corners[0][0] + tag_corners[1][0]) / 2.0
                top_mid_y = (tag_corners[0][1] + tag_corners[1][1]) / 2.0

                world_top_mid_x, world_top_mid_y = self.pixel_to_world(
                    top_mid_x, top_mid_y, W, H, Sx, Sy)

                yaw = math.atan2(world_top_mid_y - world_y,
                                 world_top_mid_x - world_x)

                # Assemble PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "map"

                pose_msg.pose.position.x = float(world_x)
                pose_msg.pose.position.y = float(world_y)
                pose_msg.pose.position.z = 0.0

                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
                pose_msg.pose.orientation.w = math.cos(yaw / 2.0)

                # Dynamically create a publisher for this specific ID if it doesn't exist yet
                if tag_id not in self.robot_pubs:
                    topic_name = f'/arena/robot_{tag_id}/pose'
                    self.robot_pubs[tag_id] = self.create_publisher(
                        PoseStamped, topic_name, 10)
                    self.get_logger().info(f"Registered new publisher for ID {
                        tag_id} at {topic_name}")

                # Publish the specific robot's pose
                self.robot_pubs[tag_id].publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArenaPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
