import keras.models
import numpy as np
import os

import pandas as pd
import pygame
from obstacles import draw_obstacles, obst, walls
import robot
import lines
import math
global color
# training variable - if training, we are controlling the robot
training = False

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

import csv

# special loop made for human-NN training
def control():
    # initialize global variables
    global signal_array
    global training
    global robot_x
    global robot_y
    global robot_rotation
    global robot_speed

    global ground_truth_action
    global write
    write = False
    # initialize other variables
    drawing_pause = False
    control = True

    columns = ["sensor_signals","action"]


    # training loop
    while control:

        for event in pygame.event.get():
            # control quiting command -------------------------------------------
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # control drawing ---------------------------------------------------
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

            # control keys -------------------------------------------------------
            if event.type == pygame.KEYDOWN:

                # ending training loop -------------------------------------------
                if event.key == pygame.K_t:
                    training = False
                    control = False

                # controlling speed ----------------------------------------------
                if event.key == pygame.K_UP:
                    robot_speed += 3
                if event.key == pygame.K_DOWN:
                    robot_speed -= 3
                    if robot_speed < 0.5:
                        robot_speed = 0.5

                # selecting an action --------------------------------------------
                if event.key == pygame.K_LEFT:
                    ground_truth_action = 2
                    robot_rotation += 3
                    write = True
                elif event.key == pygame.K_RIGHT:
                    ground_truth_action = 1
                    robot_rotation -= 3
                    write = True
                elif event.key == pygame.K_SPACE:
                    ground_truth_action = 0
                    robot_x, robot_y = robot.forward(robot_rotation, robot_x, robot_y, robot_speed)
                    write = True


                # training code here ---------------------------------------------
                if (write is True) and (signal_array !=[0,0,0,0,0,0,0]) and signal_array != []:
                    entry = np.append(signal_array,ground_truth_action)
                    # df = pd.DataFrame(entry.reshape(-1,len(entry)),columns=columns)
                    # print(df)
                    # df.to_csv("dataset.csv",mode="a",header=False)
                    from nn_tree_copy import near_sensor

                    print(signal_array)
                    try:
                        train = [([[float(i)]],np.array([near_sensor(i)])) for i in signal_array]
                    except TypeError:
                        train = [([[float(i[0])]],np.array([near_sensor(i[0])])) for i in signal_array]

                    #
                    # print(train)
                    tree_net.train(data=train,y=np.array([ground_truth_action]),train_sensors=False,epochs=2)

                    with open('dataset.csv', 'a') as data_file:
                        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                        data_writer.writerow(entry)


                # display --------------------------------------------------------
                screen.fill((150, 242, 240))
                draw_obstacles(screen)
                start, end = draw_robot(robot_rotation, robot_x, robot_y, robot_image, screen, robot_sensor_range)
                signal_array = robot.sensor_detection_efficient(robot_rotation, robot_x, robot_y, screen, robot_sensor_range)
                pygame.display.flip()
                # print(signal_array)


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


# loading model components
from nn_tree_copy import TreeNet, InfraredNet, MainNet

inf = keras.models.load_model('infr_net')
infr_net = InfraredNet(None,None,3,"infr")
infr_net.load_model(inf)

infr_net_list = [infr_net,infr_net,infr_net,infr_net,infr_net,infr_net,infr_net]

main = keras.models.load_model('main_net')
main_net = InfraredNet(None,None,3,"main")
main_net.load_model(main)

tree_net = TreeNet(sensors=infr_net_list,mainnet=main_net)

# game loop
running = True
while running:
    clock.tick(60)
    # displays the content
    pygame.display.flip()

    # next action place holder
    cmd = 0

    # controlling events --------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # controlling pause ------------------------
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused(screen)

            # controlling speed ---------------------
            if event.key == pygame.K_UP:
                robot_speed += 3

            if event.key == pygame.K_DOWN:
                robot_speed -= 3
                if robot_speed < 0.5:
                    robot_speed = 0.5

        # to enable/disable training ------------------
            if event.key == pygame.K_t:
                training = not training


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

    # training functions ---------------------------
    if training:
        cmd = control()

    # painting background---------------------------
    screen.fill((150, 242, 240))
    # drawing obstacles
    draw_obstacles(screen)

    # rendering robot
    # getting start and end points of the robot and its sensor
    start, end = draw_robot(robot_rotation, robot_x, robot_y, robot_image, screen, robot_sensor_range)

    # if sensor detects anything, turn left
    try:
        signal_array = robot.sensor_detection_efficient(robot_rotation, robot_x, robot_y, screen, robot_sensor_range)

    # if robot goes out of bound, return it to middle
    except IndexError:
        robot_x = 500
        robot_y = 400
    print(signal_array)


    # TODO: neural network tree goes here, pass signal array to the network -----------


    if signal_array == [0,0,0,0,0,0,0]:
        robot_x, robot_y = robot.forward(robot_rotation, robot_x, robot_y, robot_speed)

    else:
        try:
            signal_array = [[float(i)] for i in signal_array]
        except TypeError or IndexError:
            robot_x = 500
            robot_y == 400

        decision = tree_net.predict_class(signal_array)

        decision = decision[0]

        if decision == 0:
            robot_x, robot_y = robot.forward(robot_rotation, robot_x, robot_y, robot_speed)

        elif decision == 1:
            robot_rotation -= 3

        else:
            robot_rotation +=3

# if robot_rotation > -90:
    # robot_rotation -= 0.05
    # pygame.display.update()
