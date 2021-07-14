import shapely
import math
from shapely.geometry import LineString


# Calc the gradient 'm' of a line between p1 and p2
def calculateGradient(p1, p2):
    # Ensure that the line is not vertical
    if (p1[0] != p2[0]):
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
        return m
    else:
        return None


# Find the angle of two lines
# p1 p2 are robot points, p3 p4 are obstacle points
def calculateAngle(p1,p2,p3,p4):
    m1 = calculateGradient(p1,p2)
    m2 = calculateGradient(p3,p4)

    if m2 is None:
        return 90 - abs(math.degrees(math.atan(m1)))
    if m1 is None:
        return 90 - abs(math.degrees(math.atan(m2)))

    return math.degrees(math.atan(abs((m2-m1)/(1+m2*m1))))


# Find distance from the robot to the sensor line/obstacle intersection
def calculateDistance(robot_point,collision_point):
    x_diff = robot_point[0]-collision_point[0]
    y_diff = robot_point[1]-collision_point[1]

    return math.sqrt(math.pow(x_diff,2) + math.pow(y_diff,2))


def calculateIntersectPoint(p1,p2,p3,p4):

    line1 = LineString([p1, p2])
    line2 = LineString([p3, p4])

    int_pt = line1.intersection(line2)

    if(int_pt.type != "Point"):
        return None
    point_of_intersection = int_pt.x, int_pt.y

    return point_of_intersection


p1 = (1, 5)
p2 = (4, 7)

p3 = (4, 5)
p4 = (3, 7)

p5 = (4, 1)
p6 = (3, 3)

p7 = (3, 1)
p8 = (3, 10)

p9 = (0, 6)
p10 = (5, 6)

p11 = (472.0, 116.0)
p12 = (542.0, 116.0)


assert None != calculateIntersectPoint(p1, p2, p3, p4), "line 1 line 2 should intersect"
assert None != calculateIntersectPoint(p3, p4, p1, p2), "line 2 line 1 should intersect"
assert None == calculateIntersectPoint(p1, p2, p5, p6), "line 1 line 3 shouldn't intersect"
assert None == calculateIntersectPoint(p3, p4, p5, p6), "line 2 line 3 shouldn't intersect"
assert None != calculateIntersectPoint(p1, p2, p7, p8), "line 1 line 4 should intersect"
assert None != calculateIntersectPoint(p7, p8, p1, p2), "line 4 line 1 should intersect"
assert None != calculateIntersectPoint(p1, p2, p9, p10), "line 1 line 5 should intersect"
assert None != calculateIntersectPoint(p9, p10, p1, p2), "line 5 line 1 should intersect"
assert None != calculateIntersectPoint(p7, p8, p9, p10), "line 4 line 5 should intersect"
assert None != calculateIntersectPoint(p9, p10, p7, p8), "line 5 line 4 should intersect"
print("\nSUCCESS! All asserts passed for doLinesIntersect")

