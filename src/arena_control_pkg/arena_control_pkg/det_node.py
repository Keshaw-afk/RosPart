import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import math
from functools import partial


class FollowTheGapPlanner(Node):
    def __init__(self):
        super().__init__('follow_the_gap_planner')

        self.cmd_pubs = {}
        self.state_subs = {}

        # --- FTG Parameters ---
        # Size of the safety bubble (meters) around closest obstacle
        self.bubble_radius = 0.50
        self.safe_distance = 1.2   # How far a ray must reach to be considered an "open gap"
        self.goal_stop_radius = 1.0

        # LiDAR angles (64 rays, -pi to pi)
        self.angles = np.linspace(-math.pi, math.pi, 64)

        for robot_id in range(1, 5):
            r_name = f'robot_{robot_id}'
            self.cmd_pubs[r_name] = self.create_publisher(
                Twist, f'/arena/{r_name}/cmd_vel', 10)
            self.state_subs[r_name] = self.create_subscription(
                Float32MultiArray, f'/arena/{r_name}/rl_state',
                partial(self.state_callback, robot_name=r_name), 10)

        self.get_logger().info("Follow the Gap (FTG) Planner Active. Seeking free space.")

    def preprocess_lidar(self, ranges):
        """Sanitize the LiDAR data."""
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=20.0, neginf=0.0)
        # Cap max distance to ignore irrelevant faraway walls and focus on local clutter
        ranges = np.clip(ranges, 0.0, 5.0)
        return ranges

    def state_callback(self, msg, robot_name):
        state = np.array(msg.data)
        if len(state) < 66:
            return

        goal_dist = state[0]
        goal_rel_angle = state[1]

        # 1. Stop if goal is reached
        if goal_dist < self.goal_stop_radius:
            self.cmd_pubs[robot_name].publish(Twist())
            return

        # 2. Get and preprocess LiDAR
        raw_lidar = state[2:66]
        proc_ranges = self.preprocess_lidar(raw_lidar)

        # 3. Find the closest obstacle and create a Safety Bubble
        min_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[min_idx]

        # If we are physically scraping a wall (e.g. backed into a corner), force an emergency reverse
        if min_dist < 0.35:
            twist = Twist()
            twist.linear.x = -0.4
            twist.angular.z = 0.0
            self.cmd_pubs[robot_name].publish(twist)
            return

        # Create the bubble: Zero out any rays that fall within the bubble radius of the closest point
        # This mathematically "inflates" the obstacle to account for the robot's physical width
        bubble_angles = np.arcsin(
            min(1.0, self.bubble_radius / (min_dist + 0.01)))

        # Calculate which array indices fall inside the bubble
        angle_step = (2 * math.pi) / 64.0
        bubble_idx_span = int(bubble_angles / angle_step)

        start_idx = max(0, min_idx - bubble_idx_span)
        end_idx = min(63, min_idx + bubble_idx_span)
        # Blind the robot to the danger zone
        proc_ranges[start_idx:end_idx + 1] = 0.0

        # 4. Find the largest continuous gap (sequence of safe, non-zero rays)
        # A gap is any ray reaching further than self.safe_distance
        gap_mask = proc_ranges > self.safe_distance

        # Find contiguous blocks of True
        gaps = []
        current_gap = []
        for i, is_gap in enumerate(gap_mask):
            if is_gap:
                current_gap.append(i)
            else:
                if len(current_gap) > 0:
                    gaps.append(current_gap)
                    current_gap = []
        if len(current_gap) > 0:
            gaps.append(current_gap)

        best_v = 0.0
        best_w = 0.0

        if len(gaps) == 0:
            # NO GAPS FOUND. We are boxed in completely. Spin in place to find an exit.
            best_v = 0.0
            best_w = 1.2
        else:
            # 5. Score the gaps and pick the best one
            # Find the largest gap
            largest_gap = max(gaps, key=len)

            # Find the ray inside the largest gap that points closest to the goal
            best_ray_idx = largest_gap[0]
            min_angle_diff = float('inf')

            for idx in largest_gap:
                ray_angle = self.angles[idx]
                angle_diff = abs(
                    (ray_angle - goal_rel_angle + math.pi) % (2 * math.pi) - math.pi)

                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_ray_idx = idx

            # 6. Steer toward the chosen safe ray
            target_angle = self.angles[best_ray_idx]

            best_w = max(-1.2, min(1.2, 1.5 * target_angle))

            # Dynamic velocity: Go fast if driving straight, slow down if turning sharply into a gap
            if abs(target_angle) < 0.5:
                best_v = 0.6
            else:
                best_v = 0.2

        twist = Twist()
        twist.linear.x = float(best_v)
        twist.angular.z = float(best_w)
        self.cmd_pubs[robot_name].publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGapPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
