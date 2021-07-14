import numpy as np
import os
import pygame
from obstacles import draw_obstacles, obst
import robot
import lines
import math
global color

# initialization
pygame.init()

# setting display size
width = 1000
height = 800
screen = pygame.display.set_mode((width, height))

# display title
pygame.display.set_caption("robot simulation")

# defining clock
clock = pygame.time.Clock()

# setting icon
icon = pygame.image.load("images/robot_icon.png")
pygame.display.set_icon(icon)

# creating robot body
robot_image = pygame.image.load("images/robot_body.png")
robot_image = pygame.transform.scale(robot_image, (30, 30))
# robot_image = robot_image.get_rect()

# robot info -------------------
robot_x = 300
robot_y = 700
robot_rotation = -25
robot_sensor_range = 200
robot_speed = 0.5
# ------------------------------
# drawing variable -------------
drawing = False
new_point_1 = ()
new_point_2 = ()
# ------------------------------


def paused(screen):
    pause = True
    drawing_pause = False

    while pause:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = False

            # controlling drawing new obstacles
            if event.type == pygame.MOUSEBUTTONDOWN:
                if drawing_pause is False:
                    new_point_1 = pygame.mouse.get_pos()
                    drawing_pause = True
                else:
                    new_point_2 = pygame.mouse.get_pos()
                    obst.append((new_point_1, new_point_2))
                    drawing_pause = False
                    draw_obstacles(screen)
                    pygame.display.flip()
            # ----------------------------------------


def draw_robot(rotation, x, y, robot_image, screen, sensor_range):
    robot_image = pygame.transform.rotate(robot_image, rotation)
    rect = robot_image.get_rect()
    rect.center = (x, y)

    # screen.blit(robot_image, (x, y))
    screen.blit(robot_image, rect)

    # find where the sensor end point is
    # return robot position and end point
    x_end, y_end = robot.get_robot_end_points(rotation, x, y, sensor_range)
    return (x, y), (x_end, y_end)


# game loop
running = True
while running:
    clock.tick(60)
    # displays the content
    pygame.display.flip()

    # controlling events --------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # controlling pause
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p :
                paused(screen)

        # controlling drawing new obstacles
        if event.type == pygame.MOUSEBUTTONDOWN:
            if drawing is False:
                new_point_1 = pygame.mouse.get_pos()
                drawing = True
            else:
                new_point_2 = pygame.mouse.get_pos()
                obst.append((new_point_1,new_point_2))
                drawing = False
        # ------------------------------------------

    # painting background
    screen.fill((150, 242, 240))
    # drawing obstacles
    draw_obstacles(screen)

    # rendering robot
    # getting start and end points of the robot and its sensor
    start, end = draw_robot(robot_rotation, robot_x, robot_y, robot_image, screen, robot_sensor_range)

    # if sensor detects anything, turn left
    signal_array = robot.sensor_detection_efficient(robot_rotation, robot_x, robot_y, screen, robot_sensor_range)

    print(signal_array)

    # TODO: neural network tree goes here, pass signal array to the network
    # in this section, the robot makes some type of decision (ex. robot_rotation += 90 and move forward)
    # after making this decision, the change is applied to the global variables
    # the changed position/rotation will be drawn in the next iteration

    # updating x y value of robot to make it move

    robot_x, robot_y = robot.forward(robot_rotation, robot_x, robot_y, robot_speed)

    # if robot_rotation > -90:
    # robot_rotation -= 0.05
    # pygame.display.update()
