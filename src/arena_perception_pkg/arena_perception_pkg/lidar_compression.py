import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from functools import partial


class LidarCompressionNode(Node):
    def __init__(self):
        super().__init__('lidar_compression_node')

        # RL Downsampling Configuration
        self.original_horizontal_rays = 640
        self.vertical_channels = 16
        self.target_resolution = 64

        # Dictionaries to hold our dynamic subscribers and publishers
        self.scan_subs = {}
        self.scan_pubs = {}

        # Loop through robots 1 to 4 and set up their specific topics
        for i in range(1, 5):
            robot_name = f'robot_{i}'
            sub_topic = f'/{robot_name}/lidar/scan'
            pub_topic = f'arena/{robot_name}/scan_compressed'

            # Create a publisher for this specific robot
            self.scan_pubs[robot_name] = self.create_publisher(
                LaserScan,
                pub_topic,
                10
            )

            # Create a subscriber for this specific robot
            # Using partial() allows us to pass the 'robot_name' into the callback
            # so the callback knows which publisher to route the processed data to.
            self.scan_subs[robot_name] = self.create_subscription(
                LaserScan,
                sub_topic,
                partial(self.scan_callback, robot_name=robot_name),
                10
            )

            self.get_logger().info(
                f"Initialized compression pipeline for {robot_name}")

    def scan_callback(self, msg, robot_name):
        # 1. Convert the tuple of ranges into a NumPy array (now it's already a 1D array of 640 elements)
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Guard clause: Update the expected size to match the 640 horizontal rays
        if ranges.size != self.original_horizontal_rays:
            self.get_logger().warning(
                f"[{robot_name}] Unexpected scan size: {
                    ranges.size}. Expected {self.original_horizontal_rays}.",
                throttle_duration_sec=2.0
            )
            return

        # 2. Horizontal Downsampling for RL (Max Pooling)
        # We group the 640 rays into 64 windows of 10 rays each.
        window_size = self.original_horizontal_rays // self.target_resolution

        # Reshape to (64, 10) and take the minimum distance in each window
        # so we never miss corners or thin obstacles.
        downsampled_ranges = np.min(ranges.reshape(-1, window_size), axis=1)

        # 3. RL Sanitization: Handle Inf and NaN values
        downsampled_ranges = np.nan_to_num(
            downsampled_ranges,
            posinf=msg.range_max,
            neginf=msg.range_min
        )

        # 4. Construct the new 2D LaserScan message
        compressed_msg = LaserScan()
        compressed_msg.header = msg.header

        # Copy physics parameters, updating the angle increment for the new resolution
        compressed_msg.angle_min = msg.angle_min
        compressed_msg.angle_max = msg.angle_max
        compressed_msg.angle_increment = (
            msg.angle_max - msg.angle_min) / self.target_resolution

        compressed_msg.time_increment = msg.time_increment
        compressed_msg.scan_time = msg.scan_time
        compressed_msg.range_min = msg.range_min
        compressed_msg.range_max = msg.range_max

        compressed_msg.ranges = downsampled_ranges.tolist()

        # 5. Publish to the specific robot's compressed topic
        self.scan_pubs[robot_name].publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCompressionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
