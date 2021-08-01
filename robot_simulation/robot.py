import math
import pygame
import lines
from obstacles import obst


# calculate x,y changes to go "forward" given the absolute rotation of the robot
def forward(abs_rotation, x, y, speed):
    y -= speed * math.cos(math.radians(abs_rotation))
    x -= speed * math.sin(math.radians(abs_rotation))

    return(x, y)
    # rotation = 0 -> y+1, x+0
    # rotation = 90 -> y+0, x-1


def back(abs_rotation, x, y, speed):
    y += speed * math.cos(math.radians(abs_rotation))
    x += speed * math.sin(math.radians(abs_rotation))

    return (x, y)


def get_robot_end_points(abs_rotation, x, y, range):
    y_end = y - range * math.cos(math.radians(abs_rotation))
    x_end = x - range * math.sin(math.radians(abs_rotation))
    return x_end,y_end


def draw_sensor_line(start,end, surface):

    pygame.draw.line(surface,(255, 89, 89),start,end,1)


# calculate sensor line collision with obstacles
def sensor_detection(start, end):
    for element in obst:
        obs_start = element[0]
        obs_end = element[1]

        # check for intersection
        # finds point with minimum distance
        point = lines.calculateIntersectPoint(start, end, obs_start, obs_end)
        if point is not None:
            print("hit point", point)
            print("angle: ", lines.calculateAngle(start, end, obs_start, obs_end))
            print("distance: ", lines.calculateDistance(start, point))
            print()
            return True

    return False


# calculate all obstacles and find closest one
def sensor_detection_2(start, end):
    global min_dist
    global closest_point
    global angle
    min_dist = math.inf
    closest_point = None
    angle = 0

    for element in obst:
        obs_start = element[0]
        obs_end = element[1]

        # check for intersection
        # finds point with minimum distance

        point = lines.calculateIntersectPoint(start, end, obs_start, obs_end)
        if point is not None:
            dist = lines.calculateDistance(start, point)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                angle = lines.calculateAngle(start, end, obs_start, obs_end)

    if closest_point is not None:
        print("hit point", closest_point)
        print("angle: ", angle)
        print("distance: ", min_dist)
        print()
        return True

    return False


# sensor takes in:
# robot position
# iteration number i
# position offset - where is the sensor located on the robot
# the rotation of the sensor (w/r to the robot)
# surface - the screen
def step_sense(x,y,position_offset,rotation,surface,rng):
    #original_point = (position_offset,0)
    x = x+position_offset* math.cos(math.radians(rotation))
    y = y-position_offset* math.sin(math.radians(rotation))

    # sensing everything from pxl 15 to the range
    for i in range(15,rng):
        # calculate point in direction
        y_end = y - i * math.cos(math.radians(rotation))
        x_end = x - i * math.sin(math.radians(rotation))
        # get point position
        pxl = (int(x_end), int(y_end))
        # get color
        color_key = surface.get_at(pxl)
        color = (color_key[0],color_key[1],color_key[2])

        # if detected an obstacle
        # all obstacles has equal RGB: (1,1,1) (2,2,2) etc.
        if color[0] == color[1]:
            # find index of obstacle from color key
            obs_hit = obst[color[0]]

            # (x,y) -> robot position, pxl -> hit point
            ngl = lines.calculateAngle((x,y),pxl,obs_hit[0],obs_hit[1])
            dist = lines.calculateDistance((x,y),pxl)
            # scaling distance into 0-2.9 signal
            # signal is reduced on angled surfaces
            # detail missing: recall the sensor reading curve - when object is too close the signal dips
            return round(ngl/90*((1-dist/rng)*2.9),2)

        # if nothing detected
        else:
            surface.set_at(pxl,(255, 89, 89))
    return 0


# current detection algorithm
def sensor_detection_efficient(abs_rotation, x, y, surface, rng):

    l = step_sense(x,y,0,abs_rotation+90,surface,rng)
    tl = step_sense(x,y,0,abs_rotation+45,surface,rng)

    cl = step_sense(x,y,-5,abs_rotation,surface,rng)
    cc = step_sense(x,y,0,abs_rotation,surface,rng)
    cr = step_sense(x,y,5,abs_rotation,surface,rng)

    tr = step_sense(x,y,0,abs_rotation-45,surface,rng)
    r = step_sense(x,y,0,abs_rotation-90,surface,rng)

    signal_array = [l,tl,cl,cc,cr,tr,r]

    print(signal_array)
    return signal_array


# collision sensor
def touch_detection(abs_rotation,x,y,surface):
    s1 = step_touch(abs_rotation,x,y,surface)
    s2 = step_touch(abs_rotation+45,x,y,surface)
    s3 = step_touch(abs_rotation+90,x,y,surface)
    s4 = step_touch(abs_rotation+135,x,y,surface)
    s5 = step_touch(abs_rotation+180,x,y,surface)
    s6 = step_touch(abs_rotation-45,x,y,surface)
    s7 = step_touch(abs_rotation-90,x,y,surface)
    s8 = step_touch(abs_rotation-135,x,y,surface)

    return (s1,s2,s3,s4,s5,s6,s7,s8)


def step_touch(abs_rotation,x,y,surface):
    y_end = int(y - 12* math.cos(math.radians(abs_rotation)))
    x_end = int(x - 12* math.sin(math.radians(abs_rotation)))
    color = surface.get_at((x_end,y_end))

    if color != (150, 242, 240):
        #pygame.draw.line(surface,(24, 245, 24),(x_end,y_end),(x_end,y_end),8)
        pygame.draw.circle(surface,(24,245,24),(x_end,y_end),2,2)
        return True
    else:
        #pygame.draw.line(surface,(0,0,0),(x_end,y_end),(x_end,y_end),8)
        pygame.draw.circle(surface,(0,0,0),(x_end,y_end),2,2)
        return False
