Robot Simulation Program:


Current Supported Functions:
- press p to pause
- left click two times on two points (whether paused or not) will create an obstacle with those two end points


Info - infrared sensors:
- the robot has 7 sensors installed, three directly in front, two diagonally and one on each side. (more can be added)
- a sensor will detect an obstacle if it's in range, they do not see past obstacles.
- robot will return a float array of length 7 every iteration, representing the sensor signals.
- ex. [0, 0, 1.68, 1.7, 1.72, 1.38, 0]
- the order of sensors in the array are as follows:
    [left, left diagonal, front_left, front_middle, front_right, right diagonal, right]

- signals in the array can be any number from 0 to 2.9, 0 being nothing in sight, 2.9 being directly in front.
- a signal is reduced if the sensor line doesn't perpendicularly touch the obstacle. Reduction scales with the angle.


Purpose:
- this simulation was created to test any decision-making unit on the robot.
- the sensors' characteristics were created to mimic real infrared sensors.