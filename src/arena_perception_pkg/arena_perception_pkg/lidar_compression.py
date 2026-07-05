import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarCompressionNode(Node):
    def __init__(self):
        super().__init__('lidar_compression_node')

        # Subscribe to the raw 16x640 Gazebo LiDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/robot_1/lidar/points',  # Change this to match your actual raw topic name
            self.scan_callback,
            10
        )

        # Publish the downsampled 1x64 2D LiDAR for PyTorch
        self.scan_pub = self.create_publisher(
            LaserScan,
            '/robot_1/lidar_scan_processed',
            10
        )

        # RL Downsampling Configuration
        self.original_horizontal_rays = 640
        self.vertical_channels = 16
        self.target_resolution = 64

        self.get_logger().info("LiDAR Compression Node Initialized.")

    def scan_callback(self, msg):
        # 1. Convert the tuple of ranges into a NumPy array
        ranges = np.array(msg.ranges, dtype=np.float32)

        expected_size = self.original_horizontal_rays * self.vertical_channels

        # Guard clause: ensure the array matches the expected 16x640 shape
        if ranges.size != expected_size:
            self.get_logger().warning(f"Unexpected scan size: {ranges.size}. Expected {
                expected_size}.", throttle_duration_sec=2.0)
            return

        # 2. Reshape into (16 vertical channels, 640 horizontal rays)
        matrix = ranges.reshape(
            (self.vertical_channels, self.original_horizontal_rays))

        # 3. Vertical Compression: Find the closest obstacle in each vertical slice
        # This collapses the 16 channels down to a single 1D array of 640 rays
        compressed_ranges = np.min(matrix, axis=0)

        # 4. Horizontal Downsampling for RL (Max Pooling)
        # We group the 640 rays into 64 windows of 10 rays each.
        window_size = self.original_horizontal_rays // self.target_resolution

        # Reshape to (64, 10) and take the minimum distance in each window
        # so we never miss corners or thin obstacles.
        downsampled_ranges = np.min(
            compressed_ranges.reshape(-1, window_size), axis=1)

        # 5. RL Sanitization: Handle Inf and NaN values
        # Neural networks fail if fed infinite values. Cap them to the max range.
        downsampled_ranges = np.nan_to_num(
            downsampled_ranges,
            posinf=msg.range_max,
            neginf=msg.range_min
        )

        # 6. Construct the new 2D LaserScan message
        compressed_msg = LaserScan()
        compressed_msg.header = msg.header

        # Copy physics parameters, but update the angle increment for the new resolution
        compressed_msg.angle_min = msg.angle_min
        compressed_msg.angle_max = msg.angle_max
        compressed_msg.angle_increment = (
            msg.angle_max - msg.angle_min) / self.target_resolution

        compressed_msg.time_increment = msg.time_increment
        compressed_msg.scan_time = msg.scan_time
        compressed_msg.range_min = msg.range_min
        compressed_msg.range_max = msg.range_max

        # Convert the NumPy array back to a standard Python list
        compressed_msg.ranges = downsampled_ranges.tolist()

        # Note: If you also need intensity data for RL, you can apply the exact
        # same max-pooling logic to msg.intensities here.

        self.scan_pub.publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCompressionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
