def get_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def get_robot_closes_to_ball(ball, robots):
    closest_robot = None
    min_distance = 9999
    for robot in robots:
        distance = get_distance(ball, robot)
        if distance < min_distance and distance < 0.15:
            min_distance = distance
            closest_robot = robot
    return closest_robot