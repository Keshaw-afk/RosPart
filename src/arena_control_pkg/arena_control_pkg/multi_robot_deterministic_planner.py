import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import math
from functools import partial


class MultiRobotDeterministicPlannerNode(Node):
    def __init__(self):
        super().__init__('multi_robot_deterministic_planner_node')

        self.look_ahead_time = 0.8
        self.safety_radius = 0.35

        # 1. Generate Action Menu (Shared across all robots)
        self.action_menu = []
        for v in np.linspace(0.0, 1.0, 6):
            for w in np.linspace(-1.5, 1.5, 15):
                self.action_menu.append((v, w))

        for w in np.linspace(-1.0, 1.0, 5):
            self.action_menu.append((-0.4, w))

        self.lidar_angles = np.linspace(-math.pi, math.pi, 64)

        # 2. Setup Dictionaries for Publishers and Subscribers
        self.cmd_pubs = {}
        self.state_subs = {}

        # Loop through robots 1 to 4 to dynamically create topics
        for robot_id in range(1, 5):
            robot_name = f'robot_{robot_id}'
            
            # Setup command publisher for this specific robot
            self.cmd_pubs[robot_name] = self.create_publisher(
                Twist, f'/arena/{robot_name}/cmd_vel', 10)
            
            # Setup state subscriber for this specific robot. 
            # functools.partial injects the robot_name into the callback so we know who to calculate for.
            self.state_subs[robot_name] = self.create_subscription(
                Float32MultiArray,
                f'/arena/{robot_name}/rl_state',
                partial(self.state_callback, robot_name=robot_name),
                10
            )

        self.get_logger().info("Advanced DWA Trajectory Planner Initialized for Robots 1-4.")

    def state_callback(self, msg, robot_name):
        state = msg.data
        if len(state) < 66:
            return

        goal_dist = state[0]
        goal_rel_angle = state[1]

        # --- Goal Arrival Check ---
        # If the robot is extremely close to the goal, slam the brakes and do nothing.
        if goal_dist < 0.25:
            twist = Twist()
            self.cmd_pubs[robot_name].publish(twist)
            self.get_logger().info(f"{robot_name} Goal Reached! Holding position.", throttle_duration_sec=2.0)
            return

        lidar_ranges = np.array(state[2:])

        # Map Goal
        x_goal = goal_dist * math.cos(goal_rel_angle)
        y_goal = goal_dist * math.sin(goal_rel_angle)

        # Map LiDAR
        valid_mask = lidar_ranges < 19.5
        raw_obs_x = lidar_ranges[valid_mask] * \
            np.cos(self.lidar_angles[valid_mask])
        raw_obs_y = lidar_ranges[valid_mask] * \
            np.sin(self.lidar_angles[valid_mask])

        # --- The Goal Exclusion Mask ---
        distances_to_goal = np.sqrt(
            (raw_obs_x - x_goal)**2 + (raw_obs_y - y_goal)**2)
        not_goal_mask = distances_to_goal > 0.4

        obs_x = raw_obs_x[not_goal_mask]
        obs_y = raw_obs_y[not_goal_mask]
        obstacle_points = np.column_stack((obs_x, obs_y))

        current_clearance = float('inf')
        if len(obstacle_points) > 0:
            current_clearance = np.min(np.linalg.norm(obstacle_points, axis=1))

        effective_safety = min(self.safety_radius, max(
            0.15, current_clearance - 0.05))

        best_action = (0.0, 0.0)
        best_score = float('inf')

        for v, w in self.action_menu:
            steps = 5
            dt = self.look_ahead_time / steps

            is_safe = True
            min_dist_on_path = float('inf')
            dx, dy, dtheta = 0.0, 0.0, 0.0

            for step in range(1, steps + 1):
                t = step * dt
                if abs(w) < 0.001:
                    px = v * t
                    py = 0.0
                    ptheta = 0.0
                else:
                    radius = v / w
                    px = radius * math.sin(w * t)
                    py = radius * (1 - math.cos(w * t))
                    ptheta = w * t

                if len(obstacle_points) > 0:
                    distances = np.sqrt(
                        (obstacle_points[:, 0] - px)**2 + (obstacle_points[:, 1] - py)**2)
                    step_closest = np.min(distances)

                    if step_closest < min_dist_on_path:
                        min_dist_on_path = step_closest

                    if step_closest < effective_safety:
                        is_safe = False
                        break

            if not is_safe:
                continue

            dx, dy, dtheta = px, py, ptheta

            future_angle_to_goal = math.atan2(y_goal - dy, x_goal - dx)
            heading_error = future_angle_to_goal - dtheta
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            heading_cost = abs(heading_error)

            dist_to_goal_future = math.sqrt(
                (x_goal - dx)**2 + (y_goal - dy)**2)

            obs_cost = 1.0 / (min_dist_on_path +
                              0.01) if min_dist_on_path < float('inf') else 0.0

            # --- FIXED WEIGHTS ---
            cost = (3.0 * dist_to_goal_future) + \
                   (1.0 * heading_cost) - \
                   (0.2 * v) + \
                   (1.0 * obs_cost)

            if cost < best_score:
                best_score = cost
                best_action = (v, w)

        if best_score == float('inf'):
            best_action = (0.0, 1.0)

        twist = Twist()
        twist.linear.x = float(best_action[0])
        twist.angular.z = float(best_action[1])
        
        # Publish to the specific robot's velocity topic
        self.cmd_pubs[robot_name].publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotDeterministicPlannerNode()
    
    # rclpy.spin() handles the SingleThreadedExecutor by default, meaning 
    # it will safely process callbacks for all 4 robots sequentially as messages arrive.
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
