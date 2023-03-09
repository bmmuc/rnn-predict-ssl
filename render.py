import os
import gym
import rsoccer_gym
import numpy as np
import time
import numpy as np
import ipdb
from gym.envs.classic_control import rendering
from typing import Dict, List, Tuple
import pandas as pd
import math
import ipdb
from src.aux_idx import Aux

# COLORS RGB
BLACK = (0 / 255, 0 / 255, 0 / 255)
BG_GREEN = (20 / 255, 90 / 255, 45 / 255)
LINES_WHITE = (220 / 255, 220 / 255, 220 / 255)
ROBOT_BLACK = (25 / 255, 25 / 255, 25 / 255)
BALL_ORANGE = (253 / 255, 106 / 255, 2 / 255)
# BALL_ORANGE_GHOST =   (253 /255, 106 /255, 2   /255, 0.5)
BALL_FUTURE_REAL = (255 / 255, 205 / 255, 0 / 255, 0.3)

BALL_FUTURE_GHOST = (255 / 255, 0 / 255, 0 / 255, 0.3)

TAG_BLUE = (0 / 255, 64 / 255, 255 / 255)
TAG_YELLOW = (250 / 255, 218 / 255, 94 / 255)
TAG_GREEN = (57 / 255, 220 / 255, 20 / 255)
TAG_RED = (151 / 255, 21 / 255, 0 / 255)
TAG_PURPLE = (102 / 255, 51 / 255, 153 / 255)
TAG_PINK = (220 / 255, 0 / 255, 220 / 255)

indexes_act = Aux.is_vel
# self.indexes will be not self.indexes_act
indexes = []

for value in indexes_act:
    indexes.append(not value)


class RCGymRender:
    '''
    Rendering Class to RoboSim Simulator, based on gym classic control rendering
    '''

    def __init__(self, n_robots_blue: int = 1,
                 n_robots_yellow: int = 1,
                 field_params=None,
                 simulator: str = 'vss',
                 width: int = 950,
                 height: int = 650,
                 should_render_actual_ball=True) -> None:
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
        env = gym.make(f'SSLGoToBall-v0')
        field = env.field

        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field = field
        self.ball: rendering.Transform = None
        self.blue_robots: List[rendering.Transform] = []
        self.yellow_robots: List[rendering.Transform] = []
        self.should_render_actual_ball = should_render_actual_ball
        self.max_pos = max(6 / 2, (9 / 2)
                           + self.field.penalty_length)

        # print(self.max_pos)
        # ipdb.set_trace()
        # self.max_pos = 1.2
        self.columns = ['ball_x',
                        'ball_y',
                        'robot_blue_0_x',
                        'robot_blue_0_y',
                        'robot_blue_0_sin',
                        'robot_blue_0_cos',
                        'robot_blue_1_x',
                        'robot_blue_1_y',
                        'robot_blue_1_sin',
                        'robot_blue_1_cos',
                        'robot_blue_2_x',
                        'robot_blue_2_y',
                        'robot_blue_2_sin',
                        'robot_blue_2_cos',
                        'robot_blue_3_x',
                        'robot_blue_3_y',
                        'robot_blue_3_sin',
                        'robot_blue_3_cos',
                        'robot_blue_4_x',
                        'robot_blue_4_y',
                        'robot_blue_4_sin',
                        'robot_blue_4_cos',
                        'robot_blue_5_x',
                        'robot_blue_5_y',
                        'robot_blue_5_sin',
                        'robot_blue_5_cos',
                        'robot_yellow_0_x',
                        'robot_yellow_0_y',
                        'robot_yellow_0_sin',
                        'robot_yellow_0_cos',
                        'robot_yellow_1_x',
                        'robot_yellow_1_y',
                        'robot_yellow_1_sin',
                        'robot_yellow_1_cos',
                        'robot_yellow_2_x',
                        'robot_yellow_2_y',
                        'robot_yellow_2_sin',
                        'robot_yellow_2_cos',
                        'robot_yellow_3_x',
                        'robot_yellow_3_y',
                        'robot_yellow_3_sin',
                        'robot_yellow_3_cos',
                        'robot_yellow_4_x',
                        'robot_yellow_4_y',
                        'robot_yellow_4_sin',
                        'robot_yellow_4_cos',
                        'robot_yellow_5_x',
                        'robot_yellow_5_y',
                        'robot_yellow_5_sin',
                        'robot_yellow_5_cos',
                        ]

        # Window dimensions in pixels
        screen_width = width
        screen_height = height

        # Window margin
        margin = 0.05 if simulator == "vss" else 0.35
        # Half field width
        h_len = (9 + 2*self.field.goal_depth) / 2
        # Half field height
        h_wid = 6 / 2

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

        if simulator == "ssl":
            # add field_lines
            self._add_field_lines_ssl()
            # add robots
            self._add_ssl_robots()

        # add ball
        if (self.should_render_actual_ball):
            self._add_ball()

        self._add_ball_future_real()
        self._add_ball_future_ghost()

    def denorm_data(self,  data, colmn_name):
        id_ = colmn_name.split('_')[-1]
        find = None
        if id_ == 'x':
            data = data * 9
            find = 'x'

        elif id_ == 'y':
            data = data * 6
            find = 'y'

        elif id_ == 'sin':
            data = np.arcsin(data)
            find = 'orientation'

        # else:
        #     print('não atuando em: ', colmn_name)
        return data, find

    def __del__(self):
        self.screen.close()
        del (self.screen)
        self.screen = None

    def render_frame(self, return_rgb_array: bool = False,
                     ball_actual_x=0.0, ball_actual_y=0.0,
                     ball_true_x=0.0, ball_true_y=0.0,
                     preds=[],
                     all_true=[],
                     columns=[]) -> None:
        '''
        Draws the field, ball and players.

        Parameters
        ----------
        Frame

        Returns
        -------
        None

        '''
        # ipdb.set_trace()
        x_pos_blue = []
        y_pos_blue = []
        theta_blue = []
        # ipdb.set_trace()
        x_pos_yellow = []
        y_pos_yellow = []
        theta_yellow = []
        # self.columns = columns
        ball_pred_x = preds[0]
        ball_pred_y = preds[1]

        ball_actual_x, _ = self.denorm_data(ball_actual_x, 'ball_actual_x')
        ball_actual_y, _ = self.denorm_data(ball_actual_y, 'ball_actual_y')
        ball_true_x, _ = self.denorm_data(ball_true_x, 'ball_true_x')
        ball_true_y, _ = self.denorm_data(ball_true_y, 'ball_true_y')
        # ball_actual_x = ball_actual_x
        # ball_actual_y = ball_actual_y
        # ball_true_x = ball_true_x
        # ball_true_y = ball_true_y
        ball_pred_x, _ = self.denorm_data(ball_pred_x, 'ball_pred_x')
        ball_pred_y, _ = self.denorm_data(ball_pred_y, 'ball_pred_y')

        for i, pred in enumerate(all_true[2:]):
            col = self.columns[i + 2]
            # ipdb.set_trace()
            if 'vel' in col:
                continue

            team = None
            if 'blue' in col:
                team = 'blue'
            elif 'yellow' in col:
                team = 'yellow'

            if team != None:
                value = None
                value, find = self.denorm_data(pred, col)
                if find == 'x':
                    if team == 'blue':
                        x_pos_blue.append(value)
                    elif team == 'yellow':
                        x_pos_yellow.append(value)
                    # else:
                        # print('não atuando em: ', col)

                elif find == 'y':
                    if team == 'blue':
                        y_pos_blue.append(value)
                    elif team == 'yellow':
                        y_pos_yellow.append(value)
                    # else:
                        # print('não atuando em: ', col)

                elif find == 'orientation':
                    if team == 'blue':
                        theta_blue.append(value)
                    elif team == 'yellow':
                        theta_yellow.append(value)
                    # else:
                        # print('não atuando em: ', col)
        # ipdb.set_trace()

        if (self.should_render_actual_ball):
            self.ball.set_translation(ball_actual_x, ball_actual_y)
        # ipdb.set_trace()
        self.ball_future_real.set_translation(ball_true_x, ball_true_y)
        self.ball_future_ghost.set_translation(ball_pred_x, ball_pred_y)
        # ipdb.set_trace()
        for i in range(self.n_robots_blue):
            self.blue_robots[i].set_translation(x_pos_blue[i], y_pos_blue[i])
            self.blue_robots[i].set_rotation(theta_blue[i])

        for i in range(self.n_robots_yellow):
            self.yellow_robots[i].set_translation(
                x_pos_yellow[i], y_pos_yellow[i])
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

    def _add_field_lines_ssl(self) -> None:
        field_margin = 0.3

        # Vertical Lines X
        x_border = 9 / 2
        x_goal = x_border + self.field.goal_depth
        x_penalty = x_border - self.field.penalty_length
        x_center = 0

        # Horizontal Lines Y
        y_border = 6 / 2
        y_penalty = self.field.penalty_width / 2
        y_goal = self.field.goal_width / 2

        # add outside field borders
        field_outer_border_points = [
            (x_border+field_margin, y_border+field_margin),
            (x_border+field_margin, -y_border-field_margin),
            (-x_border-field_margin, -y_border-field_margin),
            (-x_border-field_margin, y_border+field_margin)
        ]
        field_bg = rendering.FilledPolygon(field_outer_border_points)
        field_bg.set_color(*BG_GREEN)

        outer_border = rendering.PolyLine(
            field_outer_border_points, close=True)
        outer_border.set_linewidth(self.linewidth)
        outer_border.set_color(*LINES_WHITE)

        # add field borders
        field_border_points = [
            (x_border, y_border),
            (x_border, -y_border),
            (-x_border, -y_border),
            (-x_border, y_border)
        ]
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
        goal_line_left = rendering.PolyLine(goal_line_left_points, close=False)
        goal_line_left.set_linewidth(self.linewidth)
        goal_line_left.set_color(*LINES_WHITE)

        self.screen.add_geom(field_bg)
        self.screen.add_geom(outer_border)
        self.screen.add_geom(field_border)
        self.screen.add_geom(center_line)
        self.screen.add_geom(center_circle)
        self.screen.add_geom(penalty_box_right)
        self.screen.add_geom(penalty_box_left)
        self.screen.add_geom(goal_line_right)
        self.screen.add_geom(goal_line_left)

    def _add_ssl_robots(self) -> None:
        tag_id_colors: Dict[int, Dict[int, Tuple[float, float, float,]]] = {
            0: {0: TAG_PINK, 1: TAG_GREEN, 2: TAG_PINK, 3: TAG_PINK},
            1: {0: TAG_GREEN, 1: TAG_GREEN, 2: TAG_PINK, 3: TAG_PINK},
            2: {0: TAG_GREEN, 1: TAG_GREEN, 2: TAG_PINK, 3: TAG_GREEN},
            3: {0: TAG_PINK, 1: TAG_GREEN, 2: TAG_PINK, 3: TAG_GREEN},
            4: {0: TAG_PINK, 1: TAG_PINK, 2: TAG_GREEN, 3: TAG_PINK},
            5: {0: TAG_GREEN, 1: TAG_PINK, 2: TAG_GREEN, 3: TAG_PINK},
            6: {0: TAG_GREEN, 1: TAG_PINK, 2: TAG_GREEN, 3: TAG_GREEN},
            7: {0: TAG_PINK, 1: TAG_PINK, 2: TAG_GREEN, 3: TAG_GREEN},
            8: {0: TAG_GREEN, 1: TAG_GREEN, 2: TAG_GREEN, 3: TAG_GREEN},
            9: {0: TAG_PINK, 1: TAG_PINK, 2: TAG_PINK, 3: TAG_PINK},
            10: {0: TAG_PINK, 1: TAG_GREEN, 2: TAG_GREEN, 3: TAG_PINK},
            11: {0: TAG_GREEN, 1: TAG_PINK, 2: TAG_PINK, 3: TAG_GREEN},
            12: {0: TAG_GREEN, 1: TAG_GREEN, 2: TAG_GREEN, 3: TAG_PINK},
            13: {0: TAG_GREEN, 1: TAG_PINK, 2: TAG_PINK, 3: TAG_PINK},
            14: {0: TAG_PINK, 1: TAG_GREEN, 2: TAG_GREEN, 3: TAG_GREEN},
            15: {0: TAG_PINK, 1: TAG_PINK, 2: TAG_PINK, 3: TAG_GREEN}
        }
        # Add blue robots
        for id in range(self.n_robots_blue):
            self.blue_robots.append(
                self._add_ssl_robot(team_color=TAG_BLUE,
                                    id_color=tag_id_colors[id])
            )

        # Add yellow robots
        for id in range(self.n_robots_yellow):
            self.yellow_robots.append(
                self._add_ssl_robot(team_color=TAG_YELLOW,
                                    id_color=tag_id_colors[id])
            )

    def _add_ssl_robot(self, team_color, id_color) -> rendering.Transform:
        robot_transform: rendering.Transform = rendering.Transform()

        # Robot dimensions
        robot_radius: float = self.field.rbt_radius
        distance_center_kicker: float = self.field.rbt_distance_center_kicker
        kicker_angle = 2 * np.arccos(distance_center_kicker / robot_radius)
        res = 30

        points = []
        for i in range(res + 1):
            ang = (2*np.pi - kicker_angle)*i / res
            ang += kicker_angle/2
            points.append((np.cos(ang)*robot_radius, np.sin(ang)*robot_radius))

        # Robot object
        robot = rendering.FilledPolygon(points)
        robot.set_color(*ROBOT_BLACK)
        robot.add_attr(robot_transform)

        # Team Tag
        tag_team = rendering.make_circle(0.025, filled=True)
        tag_team.set_color(*team_color)
        tag_team.add_attr(robot_transform)

        # Tag 0, upper right
        tag_0 = rendering.make_circle(0.020, filled=True)
        tag_0.set_color(*id_color[0])
        tag_0.add_attr(rendering.Transform(translation=(0.035, 0.054772)))
        tag_0.add_attr(robot_transform)

        # Tag 1, upper left
        tag_1 = rendering.make_circle(0.020, filled=True)
        tag_1.set_color(*id_color[1])
        tag_1.add_attr(rendering.Transform(translation=(-0.054772, 0.035)))
        tag_1.add_attr(robot_transform)

        # Tag 2, lower left
        tag_2 = rendering.make_circle(0.020, filled=True)
        tag_2.set_color(*id_color[2])
        tag_2.add_attr(rendering.Transform(translation=(-0.054772, -0.035)))
        tag_2.add_attr(robot_transform)

        # Tag 3, lower right
        tag_3 = rendering.make_circle(0.020, filled=True)
        tag_3.set_color(*id_color[3])
        tag_3.add_attr(rendering.Transform(translation=(0.035, -0.054772)))
        tag_3.add_attr(robot_transform)

        # Add objects to screen
        self.screen.add_geom(robot)
        self.screen.add_geom(tag_team)
        self.screen.add_geom(tag_0)
        self.screen.add_geom(tag_1)
        self.screen.add_geom(tag_2)
        self.screen.add_geom(tag_3)

        # Return the transform class to change robot position
        return robot_transform

    def _add_ball(self):
        ball_radius: float = self.field.ball_radius
        ball_transform: rendering.Transform = rendering.Transform()

        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        ball.set_color(*BALL_ORANGE)
        ball.add_attr(ball_transform)

        ball_outline: rendering.Geom = rendering.make_circle(
            ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)

        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)

        self.ball = ball_transform

    def _add_ball_future_real(self):
        ball_radius: float = self.field.ball_radius
        ball_transform: rendering.Transform = rendering.Transform()

        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        # ball.set_color(*BALL_ORANGE)
        ball._color.vec4 = BALL_FUTURE_REAL
        ball.add_attr(ball_transform)

        ball_outline: rendering.Geom = rendering.make_circle(
            ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)

        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)

        self.ball_future_real = ball_transform

    def _add_ball_future_ghost(self):
        ball_radius: float = self.field.ball_radius
        ball_transform: rendering.Transform = rendering.Transform()

        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        # ball.set_color(*BALL_ORANGE)
        ball._color.vec4 = BALL_FUTURE_GHOST
        ball.add_attr(ball_transform)

        ball_outline: rendering.Geom = rendering.make_circle(
            ball_radius*1.1, filled=False)
        ball_outline.linewidth.stroke = 1
        ball_outline.set_color(*BLACK)
        ball_outline.add_attr(ball_transform)

        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)

        self.ball_future_ghost = ball_transform


if __name__ == '__main__':
    # ipdb.set_trace()
    # data = np.load('/home/bmmuc/Documents/robocin/rnn/rnn-predict-ssl/datasets/train_data_labels.npy',
    #    allow_pickle=True)
    # def render_frame(self, return_rgb_array: bool = False,
    #              ball_actual_x=0.0, ball_actual_y=0.0,
    #              ball_true_x=0.0, ball_true_y=0.0,
    #              preds=[]) -> None:
    data = pd.read_csv(
        '/home/bmmuc/Documents/robocin/logs-unification/reader/output/2021-06-21_10-22_OMID-vs-RobôCin.csv')
    data.fillna(-99999, inplace=True)
    columns_to_get = ['ball_x',
                      'ball_y',
                      # 'ball_vel_x',
                      # 'ball_vel_y'
                      ]

    for i in range(6):
        columns_to_get.append(f'robot_blue_{i}_x')
        columns_to_get.append(f'robot_blue_{i}_y')
        columns_to_get.append(f'robot_blue_{i}_orientation')
        columns_to_get.append(f'robot_blue_{i}_vel_x')
        columns_to_get.append(f'robot_blue_{i}_vel_y')
        columns_to_get.append(f'robot_blue_{i}_vel_angular')

    for i in range(6):
        columns_to_get.append(f'robot_yellow_{i}_x')
        columns_to_get.append(f'robot_yellow_{i}_y')
        columns_to_get.append(f'robot_yellow_{i}_orientation')
        columns_to_get.append(f'robot_yellow_{i}_vel_x')
        columns_to_get.append(f'robot_yellow_{i}_vel_y')
        columns_to_get.append(f'robot_yellow_{i}_vel_angular')
    # ipdb.set_trace()
    data.drop('ref_command', inplace=True, axis=1)

    def normalize_data(data, colmn_name):
        id_ = colmn_name.split('_')[-1]

        if id_ == 'x' and 'vel' not in colmn_name:
            # print('atuando em: ', colmn_name)
            data[colmn_name] = data[colmn_name].replace(-99999, 10)
            data[colmn_name] = np.clip(data[colmn_name] / 9, -1.2, 1.2)

        elif id_ == 'y' and 'vel' not in colmn_name:
            data[colmn_name] = data[colmn_name].replace(-99999, 8)
            # print('atuando em: ', colmn_name)

            data[colmn_name] = np.clip(data[colmn_name] / 6, -1.2, 1.2)

        elif id_ == 'x' and 'vel' in colmn_name:
            data[colmn_name] = data[colmn_name].replace(-99999, 9)
            # print('atuando em: ', colmn_name)

            data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)

        elif id_ == 'y' and 'vel' in colmn_name:
            data[colmn_name] = data[colmn_name].replace(-99999, 9)
            # print('atuando em: ', colmn_name)

            data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)

        elif id_ == 'orientation' and 'vel' not in colmn_name:
            data[colmn_name] = data[colmn_name].replace(-99999, 4)
            # print('atuando em: ', colmn_name)

            data[colmn_name] = np.clip(data[colmn_name] / math.pi, -1.2, 1.2)

        elif id_ == 'angular' and 'vel' in colmn_name:
            data[colmn_name] = data[colmn_name].replace(-99999, 9)
            # print('atuando em: ', colmn_name)

            data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)
        else:
            # print('não atuando em: ', colmn_name)
            pass
        return data
    data = data[columns_to_get]
    for colmn_name in columns_to_get:
        data = normalize_data(data, colmn_name)
    data = data.values
    # ipdb.set_trace()
    render = RCGymRender(6, 6, simulator='ssl')

    for i in range(len(data)):
        render.render_frame(
            False, data[i][0], data[i][1], preds=data[i][indexes])
        # time.sleep(1 / 120)

    del render
