import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import math
from functools import partial


class APFSwarmPlanner(Node):
    def __init__(self):
        super().__init__('apf_swarm_planner_node')

        # Tunable APF Gains
        self.k_goal = 0.5       # Attraction to goal
        self.k_obs = 0.05       # Repulsion from static obstacles
        self.k_swarm = 0.3      # Repulsion from other robots

        # Physical constraints
        self.obs_repulsion_radius = 1.0
        self.swarm_spacing = 0.8
        self.cluster_radius = 0.45  # Distance to stop at around the goal
        self.max_v = 0.8
        self.max_w = 1.5

        self.lidar_angles = np.linspace(-math.pi, math.pi, 64)

        self.cmd_pubs = {}
        self.state_subs = {}
        self.pose_subs = {}

        # Store global poses of all robots: { 'robot_1': (x, y, yaw), ... }
        self.global_poses = {}

        # Setup topics for Robots 1-4
        for robot_id in range(1, 5):
            r_name = f'robot_{robot_id}'
            self.global_poses[r_name] = (0.0, 0.0, 0.0)

            self.cmd_pubs[r_name] = self.create_publisher(
                Twist, f'/arena/{r_name}/cmd_vel', 10)

            self.state_subs[r_name] = self.create_subscription(
                Float32MultiArray, f'/arena/{r_name}/rl_state',
                partial(self.state_callback, robot_name=r_name), 10)

            self.pose_subs[r_name] = self.create_subscription(
                PoseStamped, f'/arena/{r_name}/pose',
                partial(self.pose_callback, robot_name=r_name), 10)

        self.get_logger().info("APF Swarm Clustering Node Initialized.")

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def pose_callback(self, msg, robot_name):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.global_poses[robot_name] = (x, y, yaw)

    def state_callback(self, msg, robot_name):
        state = msg.data
        if len(state) < 66:
            return

        goal_dist = state[0]
        goal_rel_angle = state[1]
        lidar_ranges = np.array(state[2:])

        # --- 1. Goal Attraction ---
        if goal_dist < self.cluster_radius:
            # We reached the ring, stop pushing forward.
            F_goal_x, F_goal_y = 0.0, 0.0
            twist = Twist()  # Send zero velocity
            self.cmd_pubs[robot_name].publish(twist)
            return
        else:
            F_goal_x = self.k_goal * goal_dist * math.cos(goal_rel_angle)
            F_goal_y = self.k_goal * goal_dist * math.sin(goal_rel_angle)

        # --- 2. LiDAR Obstacle Repulsion ---
        F_obs_x, F_obs_y = 0.0, 0.0
        # Filter out the goal cylinder (assume it's the cluster_radius)
        valid_mask = (lidar_ranges < self.obs_repulsion_radius) & (
            lidar_ranges > 0.05)

        # Calculate goal cartesian to exclude it from LiDAR
        x_g = goal_dist * math.cos(goal_rel_angle)
        y_g = goal_dist * math.sin(goal_rel_angle)

        for i, r in enumerate(lidar_ranges):
            if valid_mask[i]:
                angle = self.lidar_angles[i]
                lx = r * math.cos(angle)
                ly = r * math.sin(angle)

                # If this LiDAR point is the goal itself, ignore it
                if math.sqrt((lx - x_g)**2 + (ly - y_g)**2) < 0.3:
                    continue

                force_mag = self.k_obs / (r**2)
                F_obs_x -= force_mag * math.cos(angle)
                F_obs_y -= force_mag * math.sin(angle)

        # --- 3. Swarm Repulsion (Using global pose) ---
        F_swarm_x, F_swarm_y = 0.0, 0.0
        my_x, my_y, my_yaw = self.global_poses[robot_name]

        for other_name, (ox, oy, _) in self.global_poses.items():
            if other_name == robot_name:
                continue

            dx = ox - my_x
            dy = oy - my_y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < self.swarm_spacing and dist > 0.01:
                # Calculate angle to other robot in GLOBAL frame
                global_angle_to_other = math.atan2(dy, dx)

                # Convert to LOCAL frame of this robot
                local_angle_to_other = global_angle_to_other - my_yaw

                # Normalize angle
                local_angle_to_other = (
                    local_angle_to_other + math.pi) % (2 * math.pi) - math.pi

                force_mag = self.k_swarm / (dist**2)
                F_swarm_x -= force_mag * math.cos(local_angle_to_other)
                F_swarm_y -= force_mag * math.sin(local_angle_to_other)

        # --- 4. Kinematic Mapping ---
        Total_F_x = F_goal_x + F_obs_x + F_swarm_x
        Total_F_y = F_goal_y + F_obs_y + F_swarm_y

        target_angle = math.atan2(Total_F_y, Total_F_x)
        vector_magnitude = math.sqrt(Total_F_x**2 + Total_F_y**2)

        # Control Law
        v = vector_magnitude * max(0.0, math.cos(target_angle))
        w = 1.5 * target_angle  # K_w = 1.5

        # Clamp to max velocities
        twist = Twist()
        twist.linear.x = float(np.clip(v, 0.0, self.max_v))
        twist.angular.z = float(np.clip(w, -self.max_w, self.max_w))

        self.cmd_pubs[robot_name].publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = APFSwarmPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
