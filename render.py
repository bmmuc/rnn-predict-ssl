import os
import gym
import rsoccer_gym
import numpy as np
import ipdb
from gym.envs.classic_control import rendering
from typing import Dict, List, Tuple

# COLORS RGB
BLACK =         (0   /255, 0   /255, 0   /255)
BG_GREEN =      (20  /255, 90  /255, 45  /255)
LINES_WHITE =   (220 /255, 220 /255, 220 /255)
ROBOT_BLACK =   (25  /255, 25  /255, 25  /255)
BALL_ORANGE =   (253 /255, 106 /255, 2   /255)
# BALL_ORANGE_GHOST =   (253 /255, 106 /255, 2   /255, 0.5)
BALL_FUTURE_REAL = (255 / 255, 205 / 255, 0 / 255, 0.3)

BALL_FUTURE_GHOST = (255 / 255, 0/ 255, 0 / 255, 0.3)

TAG_BLUE =      (0   /255, 64  /255, 255 /255)
TAG_YELLOW =    (250 /255, 218 /255, 94  /255)
TAG_GREEN =     (57  /255, 220 /255, 20  /255)
TAG_RED =       (151 /255, 21  /255, 0   /255)
TAG_PURPLE =    (102 /255, 51  /255, 153 /255)
TAG_PINK =      (220 /255, 0   /255, 220 /255)

class RCGymRender:
    '''
    Rendering Class to RoboSim Simulator, based on gym classic control rendering
    '''

    def __init__(self, n_robots_blue: int = 1,
                 n_robots_yellow: int = 1,
                 field_params = None,
                 simulator: str = 'vss',
                 width: int = 750,
                 height: int = 650,
                 should_render_actual_ball = True) -> None:
        '''
        Creates our View object.

        Parameters
        ----------
        n_robots_blue : int
            Number of blue robots

        n_robots_yellow : int
            Number of yellow robots

        field_params : Field
            field parameters

        simulator : str


        Returns
        -------
        None

        '''
        # ipdb.set_trace()
        if(n_robots_blue == 1):
            env = gym.make('VSSTEST-v0')
        else:
            env = gym.make('VSS-v0')
        field = env.field

        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field = field
        self.ball: rendering.Transform = None
        self.blue_robots: List[rendering.Transform] = []
        self.yellow_robots: List[rendering.Transform] = []
        self.should_render_actual_ball = should_render_actual_ball
        self.max_pos = max(self.field.width / 2, (self.field.length / 2) 
                                + self.field.penalty_length)
        # print(self.max_pos)
        # ipdb.set_trace()
        # self.max_pos = 1.2
        
        # Window dimensions in pixels
        screen_width = width
        screen_height = height

        # Window margin
        margin = 0.05 if simulator == "vss" else 0.35
        # Half field width
        h_len = (self.field.length + 2*self.field.goal_depth) / 2
        # Half field height
        h_wid = self.field.width / 2
        
        self.linewidth = 3

        # Window dimensions in meters
        self.screen_dimensions = {
            "left": -(h_len + margin),
            "right": (h_len + margin),
            "bottom": -(h_wid + margin),
            "top": (h_wid + margin)
        }

        # Init window
        self.screen = rendering.Viewer(screen_width, screen_height)

        # Set window bounds, will scale objects accordingly
        self.screen.set_bounds(**self.screen_dimensions)

        # add background
        self._add_background()

        if simulator == "vss":
            # add field_lines
            self._add_field_lines_vss()

            # add robots
            self._add_vss_robots()

        # add ball
        if(self.should_render_actual_ball):
            self._add_ball()

        self._add_ball_future_real()
        self._add_ball_future_ghost()

    def __del__(self):
        self.screen.close()
        del(self.screen)
        self.screen = None

    def render_frame(self, return_rgb_array: bool = False, 
                    ball_actual_x = 0.0, ball_actual_y = 0.0,
                    ball_true_x = 0.0, ball_true_y = 0.0,
                    ball_pred_x = 0.0, ball_pred_y = 0.0,
                    robot_blue_theta = 0.0,
                    robot2_blue_theta = 0.0,
                    robot3_blue_theta = 0.0,
                    robot_yellow_theta = 0.0,
                    robot2_yellow_theta = 0.0,
                    robot3_yellow_theta = 0.0,
                    # robot_yellow_theta = 0.0,
                    robot_actual_x = 0.0, robot_actual_y = 0.0,
                    robot2_actual_x = 0.0, robot2_actual_y = 0.0,
                    robot3_actual_x = 0.0, robot3_actual_y = 0.0,
                    robot_yellow_x = 0.0, robot_yellow_y = 0.0,
                    robot2_yellow_x = 0.0, robot2_yellow_y = 0.0,
                    robot3_yellow_x = 0.0, robot3_yellow_y = 0.0,

                    ) -> None:
        '''
        Draws the field, ball and players.

        Parameters
        ----------
        Frame

        Returns
        -------
        None

        '''       
        if self.n_robots_blue == 1:
            ball_actual_x *= self.max_pos
            ball_actual_y *= self.max_pos

            robot_actual_x *= self.max_pos
            robot_actual_y *= self.max_pos

            robot_yellow_x *= self.max_pos
            robot_yellow_y *= self.max_pos
            
            ball_true_x *= self.max_pos
            ball_true_y *= self.max_pos

            ball_pred_x *= self.max_pos
            ball_pred_y *= self.max_pos

            robot_blue_theta = np.arcsin(robot_blue_theta)
            # robot_yellow_theta = np.arcsin(robot_yellow_theta)
            # self.ball.set_translation(frame.ball.x, frame.ball.y)
            if(self.should_render_actual_ball):
                self.ball.set_translation(ball_actual_x, ball_actual_y)

            self.ball_future_real.set_translation(ball_true_x, ball_true_y)
            self.ball_future_ghost.set_translation(ball_pred_x, ball_pred_y)

            # for i, blue in enumerate(afrme.robots_blue.values()):
            #     self.blue_robots[i].set_translation(blue.x, blue.y)
            #     self.blue_robots[i].set_rotation(np.deg2rad(blue.theta))
            for i in range(self.n_robots_blue):
                self.blue_robots[i].set_translation(robot_actual_x, robot_actual_y)
                self.blue_robots[i].set_rotation(robot_blue_theta)
            for i in range(self.n_robots_yellow):
                self.yellow_robots[i].set_translation(robot_yellow_x, robot_yellow_y)
            # self.yellow_robots[i].set_rotation(robot_yellow_theta)
        
        else:
            # ipdb.set_trace()
            # print('entrei no lugar certo')
            ball_actual_x *= self.max_pos
            ball_actual_y *= self.max_pos

            robot_actual_x *= self.max_pos
            robot_actual_y *= self.max_pos
            robot_blue_theta = np.arcsin(robot_blue_theta)

            robot2_actual_x *= self.max_pos
            robot2_actual_y *= self.max_pos
            robot2_blue_theta = np.arcsin(robot2_blue_theta)

            robot3_actual_x *= self.max_pos
            robot3_actual_y *= self.max_pos
            robot3_blue_theta = np.arcsin(robot3_blue_theta)

            robot_yellow_x *= self.max_pos
            robot_yellow_y *= self.max_pos
            robot_yellow_theta = np.arcsin(robot_yellow_theta)
            
            robot2_yellow_x *= self.max_pos
            robot2_yellow_y *= self.max_pos
            robot2_yellow_theta = np.arcsin(robot2_yellow_theta)

            robot3_yellow_x *= self.max_pos
            robot3_yellow_y *= self.max_pos
            robot3_yellow_theta = np.arcsin(robot3_yellow_theta)

            ball_true_x *= self.max_pos
            ball_true_y *= self.max_pos

            ball_pred_x *= self.max_pos
            ball_pred_y *= self.max_pos


            x_pos_blue = [robot_actual_x, robot2_actual_x, robot3_actual_x]
            y_pos_blue = [robot_actual_y, robot2_actual_y, robot3_actual_y]
            theta_blue = [robot_blue_theta, robot2_blue_theta, robot3_blue_theta]

            x_pos_yellow = [robot_yellow_x, robot2_yellow_x, robot3_yellow_x]
            y_pos_yellow = [robot_yellow_y, robot2_yellow_y, robot3_yellow_y]
            theta_yellow = [robot_yellow_theta, robot2_yellow_theta, robot3_yellow_theta]

            if(self.should_render_actual_ball):
                self.ball.set_translation(ball_actual_x, ball_actual_y)

            self.ball_future_real.set_translation(ball_true_x, ball_true_y)
            self.ball_future_ghost.set_translation(ball_pred_x, ball_pred_y)

            for i in range(self.n_robots_blue):
                self.blue_robots[i].set_translation(x_pos_blue[i], y_pos_blue[i])
                self.blue_robots[i].set_rotation(theta_blue[i])

            for i in range(self.n_robots_yellow):
                self.yellow_robots[i].set_translation(x_pos_yellow[i], y_pos_yellow[i])
                self.yellow_robots[i].set_rotation(theta_yellow[i])

        return self.screen.render(return_rgb_array=return_rgb_array)

    def _add_background(self) -> None:
        back_ground = rendering.FilledPolygon([
            (self.screen_dimensions["right"], self.screen_dimensions["top"]),
            (self.screen_dimensions["right"],
             self.screen_dimensions["bottom"]),
            (self.screen_dimensions["left"], self.screen_dimensions["bottom"]),
            (self.screen_dimensions["left"], self.screen_dimensions["top"]),
        ])
        back_ground.set_color(*BLACK)
        self.screen.add_geom(back_ground)

    #----------VSS-----------#

    def _add_field_lines_vss(self) -> None:
        # Vertical Lines X
        x_border = self.field.length / 2
        x_goal = x_border + self.field.goal_depth
        x_penalty = x_border - self.field.penalty_length
        x_center = 0

        # Horizontal Lines Y
        y_border = self.field.width / 2
        y_penalty = self.field.penalty_width / 2
        y_goal = self.field.goal_width / 2
        
        # Corners Angle offset
        corner = 0.07

        # add field borders
        field_border_points = [
            (x_border-corner, y_border),
            (x_border, y_border-corner),
            (x_border, -y_border+corner),
            (x_border-corner, -y_border),
            (-x_border+corner, -y_border),
            (-x_border, -y_border+corner),
            (-x_border, y_border-corner),
            (-x_border+corner, y_border)
        ]
        field_bg = rendering.FilledPolygon(field_border_points)
        field_bg.set_color(60/255,60/255,60/255)
        
        field_border = rendering.PolyLine(field_border_points, close=True)
        field_border.set_linewidth(self.linewidth)
        field_border.set_color(*LINES_WHITE)

        # Center line and circle
        center_line = rendering.Line(
            (x_center, y_border), (x_center, -y_border))
        center_line.linewidth.stroke = self.linewidth
        center_line.set_color(*LINES_WHITE)
        center_circle = rendering.make_circle(0.2, filled=False)
        center_circle.linewidth.stroke = self.linewidth
        center_circle.set_color(*LINES_WHITE)

        # right side penalty box
        penalty_box_right_points = [
            (x_border, y_penalty),
            (x_penalty, y_penalty),
            (x_penalty, -y_penalty),
            (x_border, -y_penalty)
        ]
        penalty_box_right = rendering.PolyLine(
            penalty_box_right_points, close=False)
        penalty_box_right.set_linewidth(self.linewidth)
        penalty_box_right.set_color(*LINES_WHITE)

        # left side penalty box
        penalty_box_left_points = [
            (-x_border, y_penalty),
            (-x_penalty, y_penalty),
            (-x_penalty, -y_penalty),
            (-x_border, -y_penalty)
        ]
        penalty_box_left = rendering.PolyLine(
            penalty_box_left_points, close=False)
        penalty_box_left.set_linewidth(self.linewidth)
        penalty_box_left.set_color(*LINES_WHITE)

        # Right side goal line
        goal_line_right_points = [
            (x_border, y_goal),
            (x_goal, y_goal),
            (x_goal, -y_goal),
            (x_border, -y_goal)
        ]
        goal_bg_right = rendering.FilledPolygon(goal_line_right_points)
        goal_bg_right.set_color(60/255,60/255,60/255)
        
        goal_line_right = rendering.PolyLine(
            goal_line_right_points, close=False)
        goal_line_right.set_linewidth(self.linewidth)
        goal_line_right.set_color(*LINES_WHITE)

        # Left side goal line
        goal_line_left_points = [
            (-x_border, y_goal),
            (-x_goal, y_goal),
            (-x_goal, -y_goal),
            (-x_border, -y_goal)
        ]
        goal_bg_left = rendering.FilledPolygon(goal_line_left_points)
        goal_bg_left.set_color(60/255,60/255,60/255)
        
        goal_line_left = rendering.PolyLine(goal_line_left_points, close=False)
        goal_line_left.set_linewidth(self.linewidth)
        goal_line_left.set_color(*LINES_WHITE)

        self.screen.add_geom(field_bg)
        self.screen.add_geom(goal_bg_right)
        self.screen.add_geom(goal_bg_left)
        self.screen.add_geom(field_border)
        self.screen.add_geom(center_line)
        self.screen.add_geom(center_circle)
        self.screen.add_geom(penalty_box_right)
        self.screen.add_geom(penalty_box_left)
        self.screen.add_geom(goal_line_right)
        self.screen.add_geom(goal_line_left)

    def _add_vss_robots(self) -> None:
        tag_id_colors: Dict[int, Tuple[float, float, float]] = {
            0 : TAG_GREEN,
            1 : TAG_PURPLE,
            2 : TAG_RED
        }
        
        # Add blue robots
        for id in range(self.n_robots_blue):
            self.blue_robots.append(
                self._add_vss_robot(team_color=TAG_BLUE, id_color=tag_id_colors[id])
            )
            
        # Add yellow robots
        for id in range(self.n_robots_yellow):
            self.yellow_robots.append(
                self._add_vss_robot(team_color=TAG_YELLOW, id_color=tag_id_colors[id])
            )

    def _add_vss_robot(self, team_color, id_color) -> rendering.Transform:
        robot_transform:rendering.Transform = rendering.Transform()
        
        # Robot dimensions
        robot_x: float = self.field.rbt_radius * 2
        robot_y: float = self.field.rbt_radius * 2
        # Tag dimensions
        tag_x: float = 0.030
        tag_y: float = 0.065
        tag_x_offset: float = (0.065 / 2) / 2
        
        # Robot vertices (zero at center)
        robot_vertices: List[Tuple[float,float]]= [
            (robot_x/2, robot_y/2),
            (robot_x/2, -robot_y/2),
            (-robot_x/2, -robot_y/2),
            (-robot_x/2, robot_y/2)
        ]
        # Tag vertices (zero at center)
        tag_vertices: List[Tuple[float,float]]= [
            (tag_x/2, tag_y/2),
            (tag_x/2, -tag_y/2),
            (-tag_x/2, -tag_y/2),
            (-tag_x/2, tag_y/2)
        ]
        
        # Robot object
        robot = rendering.FilledPolygon(robot_vertices)
        robot.set_color(*ROBOT_BLACK)
        robot.add_attr(robot_transform)

        # Team tag object
        team_tag = rendering.FilledPolygon(tag_vertices)
        team_tag.set_color(*team_color)
        team_tag.add_attr(rendering.Transform(translation=(tag_x_offset, 0)))
        team_tag.add_attr(robot_transform)
        
        # Id tag object
        id_tag = rendering.FilledPolygon(tag_vertices)
        id_tag.set_color(*id_color)
        id_tag.add_attr(rendering.Transform(translation=(-tag_x_offset, 0)))
        id_tag.add_attr(robot_transform)

        # Add objects to screen
        self.screen.add_geom(robot)
        self.screen.add_geom(team_tag)
        self.screen.add_geom(id_tag)

        # Return the transform class to change robot position
        return robot_transform

    def _add_ball(self):
        ball_radius: float = self.field.ball_radius
        ball_transform:rendering.Transform = rendering.Transform()
        
        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        ball.set_color(*BALL_ORANGE)
        ball.add_attr(ball_transform)
        
        ball_outline: rendering.Geom = rendering.make_circle(ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)
        
        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)
        
        self.ball = ball_transform

    def _add_ball_future_real(self):
        ball_radius: float = self.field.ball_radius
        ball_transform:rendering.Transform = rendering.Transform()
        
        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        # ball.set_color(*BALL_ORANGE)
        ball._color.vec4 = BALL_FUTURE_REAL
        ball.add_attr(ball_transform)
        
        ball_outline: rendering.Geom = rendering.make_circle(ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)
        
        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)
        
        self.ball_future_real = ball_transform

    def _add_ball_future_ghost(self):
        ball_radius: float = self.field.ball_radius
        ball_transform:rendering.Transform = rendering.Transform()
        
        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        # ball.set_color(*BALL_ORANGE)
        ball._color.vec4 = BALL_FUTURE_GHOST
        ball.add_attr(ball_transform)
        
        ball_outline: rendering.Geom = rendering.make_circle(ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)
        
        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)
        
        self.ball_future_ghost = ball_transform