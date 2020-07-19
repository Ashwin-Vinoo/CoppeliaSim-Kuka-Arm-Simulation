# ---------------- COPPELIASIM KUKA ARM SORTING USING VISUAL SENSORS ---------------

# This code helps to control the KUKA arm in coppeliasim which is tasked with sorting boxes of different shapes
# using visual sensors (Camera and RGBD)
# Created by Ashwin Vinoo - 5/2/2020

# Please refer https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm

# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.

# ----- Importing all the necessary modules -----
from coppeliasim_client import CoppeliaSimClient
from concurrent.futures import ThreadPoolExecutor
from skimage import color
from skimage.draw import line_aa
from pickle import load as pkl_load
from warnings import warn
from copy import deepcopy
from math import sin, cos, tan, atan
import tinyik
import numpy as np
import time
import sys
import cv2


# ----- Control Variables -----

# This variable holds the status of each conveyor belt. Program will end when everything is empty of objects
conveyor_loaded_status = [False, False, False, False, False, True, True]
# This variable indicates to the kuka arm thread that an object is ready for pickup
robot_pickup_ready = False
# This variable holds the coordinates that the kuka arm can use to pick up objects and identify its color
arm_pickup_point_and_object_color = []
# We want parallel processes to be running to handle conveyor related events
conveyor_thread_pool = ThreadPoolExecutor(max_workers=8)
# We create a list to store the threads created for each conveyor belt action
conveyor_threads = [None] * 8
# This flag indicates if it is the first round for the pusher head
pusher_head_first_round = True

# ----- Kuka Arm Parameters -----

# We define the height of the ROBOTIQ 85mm Gripper
gripper_height = 0.09
# We define the height of the media flange upon which the Gripper is inserted into the robot
media_flange_height = 0.029
# We define the kuka arm lengths in a dictionary
kuka_arm_length_dict = {'kuka_length_1': 0.36, 'kuka_length_2': 0.42, 'kuka_length_3': 0.4, 'kuka_length_4': 0.126}
# We define the arm angle limits in another dictionary
kuka_joint_range_dict = {'kuka_joint_1_range': [-170*np.pi/180, 170*np.pi/180],
                         'kuka_joint_2_range': [-120*np.pi/180, 120*np.pi/180],
                         'kuka_joint_3_range': [-170*np.pi/180, 170*np.pi/180],
                         'kuka_joint_4_range': [-120*np.pi/180, 120*np.pi/180],
                         'kuka_joint_5_range': [-170*np.pi/180, 170*np.pi/180],
                         'kuka_joint_6_range': [-120*np.pi/180, 120*np.pi/180],
                         'kuka_joint_7_range': [-175*np.pi/180, 175*np.pi/180]}

# ----- Launching the coppeliasim python client -----

# We create an object of the coppeliasim client class
coppeliasim_client = CoppeliaSimClient()
print('Attempting to launch python coppeliasim client')
# We attempt to connect to the coppeliasim server
connection_status = coppeliasim_client.connect_to_server()
# We check if the connection is not established
if not connection_status:
    print('Failed to connect to coppeliasim server')
    # We exit the python code execution as there was an error
    sys.exit("Aborting code execution")
# We inform the user that we have connected successfully to the coppeliasim server
print('Connected to the coppeliasim server')

# ----- Obtaining the handles of all the necessary sensors and actuators -----

# We obtain the list of errors and handles associated with the objects passed to the server
object_error_list, object_handle_list = coppeliasim_client.get_object_handle(['kinect_rgb', 'kinect_depth'])
# We check if all values are without error. Otherwise it indicates that some objects weren't identified
if any(object_error_list):
    print('Some objects do not have handles. Check the coppeliasim scene')
    # We exit the python code execution as there was an error
    sys.exit("Aborting code execution")
# The handles of all individual objects are separated out into their own variables for easy usability
handle_kinect_rgb, handle_kinect_depth = object_handle_list
# We obtain the list of errors and handles associated with the objects passed to the server
proximity_sensor_error_list, proximity_sensor_handle_list = coppeliasim_client.get_object_handle(['pusher_head_sensor',
                                                                                                 'kuka_gripper_sensor'])
# We check if all values are without error. Otherwise it indicates that some objects weren't identified
if any(proximity_sensor_error_list):
    print('Some objects do not have handles. Check the coppeliasim scene')
    # We exit the python code execution as there was an error
    sys.exit("Aborting code execution")
# We obtain the list of errors and handles associated with the 7 kuka arm joints
joint_error_list, multiple_joint_handles = coppeliasim_client.get_object_handle(['kuka_joint_1', 'kuka_joint_2',
                                                                                 'kuka_joint_3', 'kuka_joint_4',
                                                                                 'kuka_joint_5', 'kuka_joint_6',
                                                                                 'kuka_joint_7', 'ROBOTIQ_85_active1',
                                                                                 'ROBOTIQ_85_active2',
                                                                                 'pusher_head_joint'])
# We check if all values are without error. Otherwise it indicates that some objects weren't identified
if any(joint_error_list):
    print('Some kuka arm joints do not have handles. Check the coppeliasim scene')
    # We exit the python code execution as there was an error
    sys.exit("Aborting code execution")

# ----- Retrieving the kinect background image and depth buffer along with other image control variables -----

# We retrieve the kinect rgb image and depth buffer resolutions
rgb_resolution, rgb_background, grayscale_resolution, \
    grayscale_background, depth_resolution, depth_background = pkl_load(open("kinect_background.p", "rb"))
# This variable helps control the source of the image buffer for the live video stream
image_buffer_lock = False
# This variable helps inform code whether the images were recently updated
image_update_flag = True
# This kernel controls the extent of erosion of the threshold image
erosion_kernel = np.ones((5, 5), np.uint8)
# We use these terms to define the transformation matrix below
kinect_cos_theta = cos(65/180.0 * np.pi)
kinect_sin_theta = sin(65/180.0 * np.pi)
# We define the transformation matrix that will convert a point from the kinect coordinate frame to the kuka robots one
kinect_transformation_matrix = [[0, -kinect_sin_theta, -kinect_cos_theta, 0.9583], [-1, 0, 0, -0.0141],
                                [0, kinect_cos_theta, -kinect_sin_theta, 0.4872], [0, 0, 0, 1]]
# We create a color dictionary for reference
color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'pink', 5: 'turquoise'}

# ----- Functions to control conveyor belt operations -----


# This function helps move items onto the final collection bins from conveyors 1,2,3
def move_onto_collection_bin(conveyor_number):
    # We specify that we are using the global conveyor load status
    global conveyor_loaded_status, latest_simulation_time
    # we check if the user entered a bin number that is outside 1,2,3
    if move_onto_collection_bin not in [1, 2, 3]:
        warn('Warning: Conveyor belt chosen is outside the scope of usability for this function')
    # we check if the conveyor is not loaded
    if not conveyor_loaded_status[conveyor_number - 1]:
        warn('Warning: Conveyor belt ' + str(conveyor_number) + ' is not loaded')
    # We move the conveyor forward at 0.4m/s for 5s
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_' + str(conveyor_number)], [0.4])
    # We record the current simulation time as the start time
    conveyor_start_time = deepcopy(latest_simulation_time)
    # We delay until we are sure the conveyor covered the required distance
    while abs(latest_simulation_time - conveyor_start_time) < (4 if conveyor_number == 2 else 5.35):
        # We wait for 0.1 seconds before checking again
        time.sleep(0.1)
    # We set the conveyor speed back to zero
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_' + str(conveyor_number)], [0.0])
    # We indicate that the conveyor is no longer loaded
    conveyor_loaded_status[conveyor_number-1] = False


# This function helps load an object from conveyor 5 onto conveyor 4
def load_onto_conveyor_4():
    # We specify that we are using the global conveyor load status
    global conveyor_loaded_status, latest_simulation_time
    # we check if conveyor belt 4 is already loaded
    if conveyor_loaded_status[3]:
        warn('Warning: Conveyor belt 4 is already loaded')
    # We move the conveyor forward at 0.5m/s for 3s
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_5'], [0.3])
    # We delay until we are sure the conveyor covered the required distance of 1.57m
    # We record the current simulation time as the start time
    conveyor_5_start_time = deepcopy(latest_simulation_time)
    # We delay until we are sure the conveyor covered the required distance of 2m
    while abs(latest_simulation_time - conveyor_5_start_time) < 5.4:
        # We wait for 0.1 seconds before checking again
        time.sleep(0.1)
    # We set the conveyor speed back to zero
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_5'], [0.0])
    # We indicate that conveyor 5 is no longer loaded
    conveyor_loaded_status[4] = False
    # We indicate that conveyor 4 is now loaded
    conveyor_loaded_status[3] = True


# This function is used to move to next shape on conveyor 6 for loading onto conveyor 5
def load_onto_conveyor_5():
    # We specify that we are using the global conveyor load status
    global conveyor_loaded_status, latest_simulation_time, multiple_joint_handles, \
        latest_multiple_joint_angles, pusher_head_first_round
    # we check if conveyor belt 5 is already loaded
    if conveyor_loaded_status[4]:
        warn('Warning: Conveyor belt 5 is already loaded')
    # We check if it is the first pusher head round
    if pusher_head_first_round:
        # We move to the next object if the pusher head isn't already in front of one
        move_conveyor_6_till_next_object()
        # We mark that it is no longer the first round
        pusher_head_first_round = False
    # We start a timer
    pusher_head_start_time = deepcopy(latest_simulation_time)
    # We give the command to move the pusher head forward to push the objects onto conveyor 5
    coppeliasim_client.control_joint_property([multiple_joint_handles[9]], [0.2], 'target_position')
    # We run a while loop that waits for the pusher head to reach ~0.2m
    while True:
        # We wait for 0.1 seconds
        time.sleep(0.1)
        # We check if the pusher head position is greater than 0.19m
        if latest_multiple_joint_angles[9] > 0.19:
            # The pusher head has successfully pushed the objects down and thus we break the while loop
            break
        # In some rare cases the pusher head can get jammed
        if abs(latest_simulation_time - pusher_head_start_time) > 3:
            # We give the command to move the pusher head back to its initial position
            coppeliasim_client.control_joint_property([multiple_joint_handles[9]], [0.0], 'target_position')
            # We run a while loop that waits for the pusher head to reach ~0.0m
            while True:
                # We check if the pusher head position has reached its initial position
                if latest_multiple_joint_angles[9] < 0.01:
                    # The pusher head has successfully reached its original position and thus we break the while loop
                    break
                # We sleep for 100ms before checking the pusher head status again
                time.sleep(0.1)
            # We move conveyor 6 forward at 0.3m/s
            coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_6'], [0.3])
            # We check the time at which we started deviating from the point at which we got stuck
            deviation_start_time = deepcopy(latest_simulation_time)
            # We want conveyor belt 6 to deviate for at least one second to escape the point we got stuck
            while abs(latest_simulation_time - deviation_start_time) > 1.0:
                # We sleep for 100ms before checking the pusher head status again
                time.sleep(0.1)
            # We move to the next object if the pusher head isn't already in front of one
            move_conveyor_6_till_next_object()
            # We break out of this function
            return
        # We sleep for 100ms before checking the pusher head status again
        time.sleep(0.1)
    # We indicate that conveyor 5 is now loaded
    conveyor_loaded_status[4] = True
    # We give the command to move the pusher head back to its initial position
    coppeliasim_client.control_joint_property([multiple_joint_handles[9]], [0.0], 'target_position')
    # We run a while loop that waits for the pusher head to reach ~0.2m
    while True:
        # We check if the pusher head position has reached its initial position
        if latest_multiple_joint_angles[9] < 0.01:
            # The pusher head has successfully reached its original position and thus we break the while loop
            break
        # We sleep for 100ms before checking the pusher head status again
        time.sleep(0.1)
    # We move to the next object if the pusher head isn't already in front of one
    move_conveyor_6_till_next_object()


# This function helps move the next object on conveyor 6 to the front of the pusher head
def move_conveyor_6_till_next_object():
    # We specify that we are using the global conveyor load status
    global latest_proximity_state_list, conveyor_loaded_status, latest_simulation_time
    # we check if conveyor belt 6 is already loaded
    if not conveyor_loaded_status[5]:
        warn('Warning: This function is called when conveyor 6 is identified as empty')
    # We check if no object is already in front of the pusher head and there is still objects on conveyor 6
    if not latest_proximity_state_list[0] and conveyor_loaded_status[5]:
        # We move conveyor 6 forward at 0.3m/s
        coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_6'], [0.3])
        # We start a timer
        conveyor_6_start_time = deepcopy(latest_simulation_time)
        # We run a while loop until we detect an object in front of pusher head sensor
        while not latest_proximity_state_list[0]:
            # We timeout after 18 seconds
            if abs(latest_simulation_time - conveyor_6_start_time) > 18:
                # We specify that conveyor 6 is not loaded
                conveyor_loaded_status[5] = False
                # we break the while loop
                break
            # We sleep for 0.05 seconds before moving onto the next cycle
            time.sleep(0.05)
        # We move tell conveyor belt 6 to stop
        coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_6'], [0])


# This function moves conveyor 7 at a slow pace until we cover its length
def move_conveyor_7_till_end():
    # We specify that we are using the global conveyor load status
    global conveyor_loaded_status, latest_simulation_time
    # we check if conveyor belt 7 is already loaded
    if not conveyor_loaded_status[5]:
        warn('Warning: This function is called when conveyor 7 is identified as empty')
    # We move the conveyor forward at 0.01m/s
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_7'], [0.01])
    # We start a timer
    conveyor_7_start_time = deepcopy(latest_simulation_time)
    # We wait till conveyor 7 completes its distance
    while abs(latest_simulation_time - conveyor_7_start_time) < 157:
        # We wait around and don't do anything
        time.sleep(0.1)
    # Now that we are out of the loop, we stop moving the conveyor 7
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_7'], [0.0])
    # We specify that conveyor 7 is no longer loaded
    conveyor_loaded_status[6] = False


# This function moves conveyor 4 forward until an object has been detected or its length has been covered
def move_conveyor_4_until_object_identified():
    # We specify that we are using the global conveyor load status
    global conveyor_loaded_status, latest_simulation_time, robot_pickup_ready, arm_pickup_point_and_object_color,\
         image_buffer, image_buffer_lock, latest_rgb_image, latest_depth_buffer, image_update_flag
    # We check if conveyor 4 is loaded
    if not conveyor_loaded_status[3]:
        # We warn the user that conveyor 4 is not loaded
        warn('Warning: Conveyor belt 4 is empty and thus kinect will not scan anything')
    # We move the conveyor forward at 0.3m/s
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_4'], [0.3])
    # We record the current simulation time as the start time
    conveyor_4_start_time = deepcopy(latest_simulation_time)
    # This variable holds the total amount of time that conveyor 4 was paused
    conveyor_pause_time = 0
    # We run the while loop until we covered the entire 1.6m length of conveyor
    while abs(latest_simulation_time - conveyor_4_start_time) < 5.3 + conveyor_pause_time:
        # We find the list of acceptable contours
        contours, _, contour_count, _ = find_contours(latest_rgb_image, latest_depth_buffer)
        # Now we find the acceptable contours out of those in the list
        filtered_contours_list, filtered_contour_count = find_acceptable_contours(contours)
        # We check if the number of contours is greater than zero
        if filtered_contour_count > 0:
            # We set conveyor 4 to stop moving as we have covered the entire conveyor belt
            coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_4'], [0.0])
            # We want this function to obtain control of the image buffer for the live video feed
            image_buffer_lock = True
            # We wait for 0.6 seconds to give the conveyor belt sufficient time to settle
            while coppeliasim_client.get_conveyor_4_speed() != 0.0:
                # We sleep for 0.1 seconds
                time.sleep(0.1)
            # We record the time at which we stopped the conveyor belt
            conveyor_4_stop_time = deepcopy(latest_simulation_time)
            # We wait till conveyor 7 completes its distance
            while abs(latest_simulation_time - conveyor_4_stop_time) < 0.5:
                # We sleep for 0.1 seconds
                time.sleep(0.1)
            # We set the image update flag as false and wait for it to turn true
            image_update_flag = False
            # We run a while loop until the flag turns positive
            while not image_update_flag:
                # We sleep for 0.1 seconds
                time.sleep(0.1)
            # We find the list of contours detected
            contours, contour_center_list, contour_count, contour_length = find_contours(latest_rgb_image,
                                                                                         latest_depth_buffer)
            # Now we find the acceptable contours out of those in the list
            filtered_contours_list, filtered_contour_count = find_acceptable_contours(contours)
            # We check if we have at least one acceptable contour
            while filtered_contour_count > 0:
                # We obtain the list of filtered contours
                filtered_contours = np.take(contours, filtered_contours_list, axis=0)
                # We obtain the centers of the acceptable contours
                filtered_contour_center_list = np.take(contour_center_list, filtered_contours_list, axis=0)
                # We use this variable to store the index of the selected contour
                selected_contour = 0
                # We use this variable to identify the contour that is on top
                contour_minimum_depth = filtered_contour_center_list[0][2]
                # We iterate through the filtered contour list
                for j in range(1, filtered_contour_count):
                    # We check if the contour height is greater than the maximum height recorded
                    if filtered_contour_center_list[j][2] < contour_minimum_depth:
                        # We update the maximum recording
                        contour_minimum_depth = filtered_contour_center_list[j][2]
                        # We select this contour for initial pickup
                        selected_contour = j
                # We obtain the center of the contour in terms of kinect row, column, depth
                pickup_contour_center = filtered_contour_center_list[selected_contour]
                # We obtain the coordinates to grasp the object in terms of kuka base frame
                kuka_x, kuka_y, kuka_z = convert_kinect_coordinates_to_kuka_frame_coordinates(pickup_contour_center)
                # We find the orientation to pickup the object
                pickup_orientation = find_pickup_orientation(filtered_contours_list[selected_contour], contours,
                                                             contour_center_list)
                # We obtain the contour color list of the filtered contours
                filtered_contour_color_list = find_contour_colors(latest_rgb_image, filtered_contour_center_list)
                # We obtain a grayscale image with three channels
                grayscale_image_with_3_channels = color.gray2rgb(color.rgb2gray(latest_rgb_image))
                # We mark the contour color on the grayscale image with 3 channels
                image_buffer = mark_contour_colors_bgr(grayscale_image_with_3_channels, filtered_contours,
                                                       filtered_contour_center_list, filtered_contour_color_list,
                                                       pickup_contour_center, pickup_orientation)
                # We print the results for easy viewing
                print('Kuka Pickup Coordinates - X: ' + str(kuka_x) + ' metres, Y: ' + str(kuka_y) + ' metres, Z: ' +
                      str(kuka_z) + ' metres, Orientation: ' + str(pickup_orientation*180.0/np.pi) +
                      ' degrees, Color: ' + color_dict[filtered_contour_color_list[0]])
                # We specify the point at which the robotic arm should pick up the object
                arm_pickup_point_and_object_color = [kuka_x, kuka_y, kuka_z, pickup_orientation,
                                                     filtered_contour_color_list[selected_contour]]
                # We set the robot loaded status as true to tell the kuka arm that there is an object to pick up
                robot_pickup_ready = True
                # We do not move conveyor 4 while an item is to be placed for pickup by the arm
                while robot_pickup_ready:
                    # We sleep for 0.1 seconds
                    time.sleep(0.1)
                # We record the time at which the robot picked up the object
                object_pickup_time = deepcopy(latest_simulation_time)
                # We wait till everything stabilizes
                while abs(latest_simulation_time - object_pickup_time) < 0.25:
                    # We sleep for 0.1 seconds
                    time.sleep(0.1)
                # We set the image update flag as false and wait for it to turn true
                image_update_flag = False
                # We run a while loop until the flag turns positive
                while not image_update_flag:
                    # We sleep for 0.1 seconds
                    time.sleep(0.1)
                # We find the list of contours detected
                contours, contour_center_list, contour_count, contour_length = find_contours(latest_rgb_image,
                                                                                             latest_depth_buffer)
                # Now we find the acceptable contours out of those in the list
                filtered_contours_list, filtered_contour_count = find_acceptable_contours(contours)
            # We move the conveyor forward at 0.3m/s
            coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_4'], [0.3])
            # We update the conveyor pause time to account for the time the arm spent to pick up the object(s)
            conveyor_pause_time += abs(latest_simulation_time - conveyor_4_stop_time)
        # We no longer want this function to obtain control of the image buffer for the live video feed
        image_buffer_lock = False
        # We pause before executing the next cycle
        time.sleep(0.1)
    # We set conveyor 4 to stop moving as we have covered the entire conveyor belt
    coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_4'], [0.0])
    # We specify that there are no more items for the arm to pick up and thus conveyor 4 is empty
    conveyor_loaded_status[3] = False

# ----- Functions that facilitate image processing of the object on conveyor belt 4 -----


# This function returns the orientation at which we must pickup the object
def find_pickup_orientation(pickup_contour_index, contours, contour_center_list):
    # We obtain the pickup point center
    pickup_contour_center = contour_center_list[pickup_contour_index]
    # We obtain the current contour
    pickup_contour = contours[pickup_contour_index]
    # We obtain the length of the current contour
    pickup_contour_length = len(pickup_contour)
    # This variable will store the point closest to the contour center
    closest_point = [pickup_contour[0][0][1], pickup_contour[0][0][0]]
    # This variable will hold the distance between the closest point and the contour center
    closest_distance = (pickup_contour_center[0] - pickup_contour[0][0][1])**2 + \
                       (pickup_contour_center[1] - pickup_contour[0][0][0])**2
    # We iterate through all the contour elements
    for j in range(1, pickup_contour_length):
        # We obtain the current distance of the point to the contour center
        current_distance = (pickup_contour_center[0] - pickup_contour[j][0][1])**2 + \
                        (pickup_contour_center[1] - pickup_contour[j][0][0])**2
        # We check if the current contour point is closer than our closest point recorded
        if closest_distance > current_distance:
            # We update the closest_point
            closest_point = [pickup_contour[j][0][1], pickup_contour[j][0][0]]
            # We update the closest distance
            closest_distance = current_distance
    # We update the closest distance by taking square root
    closest_distance **= 0.5
    # We check if the rows are identical
    if closest_point[0] == pickup_contour_center[0]:
        # In this case the pickup angle is 90 degrees
        pickup_orientation = 90
    # We use slope formula in this case
    else:
        # We calculate the slope = c2-c1/r2-r1 to match the X-Y axis of the kuka frame
        slope = (closest_point[1] - pickup_contour_center[1])/float(closest_point[0] - pickup_contour_center[0])
        # We obtain the pickup orientation
        pickup_orientation = atan(slope)
    # We check if we have more than one contours
    if len(contours) > 1:
        # We use this variable to store the orientation between the pickup contour center and that of other contours
        average_center_orientation = None
        # We use this term to count the number of contours considered
        contours_considered = 0.0
        # We iterate through the other contours
        for j in range(len(contours)):
            # We check if we are not on the pickup contour and if the distances between the centers are close
            if j != pickup_contour_index and \
                ((contour_center_list[pickup_contour_index][0] - contour_center_list[j][0])**2 +
                    (contour_center_list[pickup_contour_index][1] - contour_center_list[j][1])**2)**0.5 < 125:
                # We obtain the slopes between the centers
                slope = (contour_center_list[pickup_contour_index][1] -
                         contour_center_list[j][1])/float(contour_center_list[pickup_contour_index][0] -
                                                          contour_center_list[j][0])
                # We check if we haven't updated the average value before
                if average_center_orientation is None:
                    # We initialize it with the current slope
                    average_center_orientation = atan(slope)
                else:
                    # We add the slopes  between centers of the pickup contour and the current contour
                    average_center_orientation += atan(slope)
                # We increment the count by one
                contours_considered += 1
        # We check if the average_center_orientation has been initialized and it is close to current pickup orientation
        if average_center_orientation is not None and abs(pickup_orientation -
                                                          average_center_orientation / contours_considered) < np.pi/4:
            # We offset the pickup orientation by 90 degrees to account for the blocking obstacle
            pickup_orientation += np.pi/2.0 * (-1 if pickup_orientation > 0 else 1)
    # We return the pickup_orientation
    return pickup_orientation


# This function converts the row, column and depth information to the robotic arms x, y, z
def convert_kinect_coordinates_to_kuka_frame_coordinates(kinect_coordinates):
    # We define the global variables we will use in this function
    global kinect_transformation_matrix, depth_resolution
    # The scaling factor for kinect x-axis uses the fov of 57 degrees we set in coppeliasim
    kinect_scale_x = 2 * tan(57 * np.pi / 360.0)
    # However the scaling factor for kinect y-axis uses a fov which is reduced as follows
    kinect_scale_y = 2 * tan(57 * depth_resolution[0] * np.pi / (360.0 * depth_resolution[1]))
    # We extract the kinect coordinates
    kinect_row, kinect_column, kinect_depth = kinect_coordinates
    # We convert the kinect depth to kinect z-coordinate in metres
    kinect_z = kinect_depth * 0.899 + 0.001
    # We convert the kinect column to kinect x-coordinate in metres
    kinect_x = -kinect_z * kinect_scale_x * (kinect_column/float(depth_resolution[1]) - 0.5)
    # We convert the kinect row to kinect y-coordinate in metres
    kinect_y = -kinect_z * kinect_scale_y * (kinect_row/float(depth_resolution[0]) - 0.5)
    # We transform the coordinates from the kinect frame to the kuka base frame
    kuka_coordinates = np.matmul(kinect_transformation_matrix, [[kinect_x], [kinect_y], [kinect_z], [1]])
    # We return the obtained coordinates
    return [kuka_coordinates[0][0], kuka_coordinates[1][0], kuka_coordinates[2][0]]


# This function helps identify contours and their properties
def find_contours(rgb_image, depth_buffer):
    # We specify the global variables used
    global depth_resolution, depth_background, erosion_kernel
    # We create a threshold image which is initialized to zeros
    threshold_image = np.zeros(depth_resolution, 'uint8')
    # We apply a threshold to find the change in depth images
    threshold_image[depth_background - depth_buffer > 0.03] = 255
    # We iterate through the rows
    for j in range(1, depth_resolution[0], 2):
        # We iterate through the columns
        for k in range(1, depth_resolution[1], 2):
            # We check if we are on a part of an identified object
            if threshold_image[j, k] == 255:
                # We extract the pixels color
                pixel_color_group = find_pixel_color(rgb_image[j, k])
                # This flag helps identify if a color mismatch occurred
                color_mismatch_identified = False
                # We iterate through the row offsets
                for m in range(-1, 1):
                    # We iterate through the column offsets
                    for n in range(-1, 1):
                        # We check if the color groups do not match
                        if (m != 0 or n != 0) and threshold_image[j+m, k+n] > 0 and \
                                        pixel_color_group != find_pixel_color(rgb_image[j+m, k+n]):
                            # We specify that a color mismatch was identified
                            color_mismatch_identified = True
                            # We break out of the inner for loop
                            break
                    # We break the outer for loop if color mismatch has been identified
                    if color_mismatch_identified:
                        # We break out of the outer for loop
                        break
                # We check if a color mismatch was identified
                if color_mismatch_identified:
                    # We iterate through the row offsets
                    for m in range(-1, 1):
                        # We iterate through the column offsets
                        for n in range(-1, 1):
                            # We remove that pixel from the threshold mask
                            threshold_image[j + m, k + n] = 0
    # We obtain the eroded image. Erosion helps separate two objects of same color that might have touched at a point
    threshold_image = cv2.erode(threshold_image, erosion_kernel, iterations=1)
    # We obtain the contours and its hierarchy
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # We identify the number of contours found
    contour_count = len(contours)
    # We use this to store the length of each contour
    contour_length = []
    # We create this list to store the central coordinates of each contour
    contour_center_list = []
    # We iterate through the contours
    for j in range(contour_count):
        # We update the contours length
        contour_length.append(len(contours[j]))
        # We initialize a variable to hold the contour centers
        contour_center = [0, 0]
        # We iterate through each element of the contours
        for k in range(contour_length[j]):
            # We update the row element of the contour center list
            contour_center[0] += contours[j][k][0][1]
            # We update the column element of the contour center list
            contour_center[1] += contours[j][k][0][0]
        # We now take the average to get the central row of that contour
        contour_row_center = contour_center[0]/float(contour_length[j])
        # We now take the average to get the central column of that contour
        contour_column_center = contour_center[1]/float(contour_length[j])
        # We now append the center coordinate to the contour center list
        contour_center_list.append([contour_row_center, contour_column_center,
                                    depth_buffer[int(contour_row_center), int(contour_column_center)]])
    # We return the obtained contours and their properties
    return contours, contour_center_list, contour_count, contour_length


# This function helps identify contours that are acceptable for being picked up by the kuka arm
def find_acceptable_contours(contours):
    # We create a list to store the indices of contours we have decided to capture
    filtered_contours_list = []
    # We iterate through the contour elements and search if any one can be captured
    for j in range(len(contours)):
        # We obtain the length of the contour
        contour_length = len(contours[j])
        # We initialize the leftmost contour column element
        leftmost_contour_column = depth_resolution[1]
        # We initialize the rightmost contour column element
        rightmost_contour_column = 0
        # We iterate through the contour elements
        for k in range(contour_length):
            # We check if the current contour element column is more left
            if contours[j][k][0][0] < leftmost_contour_column:
                # We update the leftmost contour element
                leftmost_contour_column = contours[j][k][0][0]
            # We check if the current contour element column is more right
            if contours[j][k][0][0] > rightmost_contour_column:
                # We update the leftmost contour element
                rightmost_contour_column = contours[j][k][0][0]
        # We check if the elements of the contour are within the specified limits and contour size isn't too small
        if rightmost_contour_column < depth_resolution[1] * 4.0/5.0 and len(contours[j]) > 75:
            # We append the current index onto the selected contour list
            filtered_contours_list.append(j)
    # We identify the number of acceptable contours
    filtered_contour_count = len(filtered_contours_list)
    # We return the number of acceptable contours and a list containing their elements
    return filtered_contours_list, filtered_contour_count


# This function captures the current depth buffer and rgb image
def capture_rgb_image_and_depth_buffer():
    # We specify that we are using the global conveyor load status
    global handle_kinect_depth, handle_kinect_rgb
    # We obtain the depth image from the kinect in coppeliasim
    _, _, depth_buffer = coppeliasim_client.read_depth_buffer_from_sensors([handle_kinect_depth])
    # We separate the numpy matrix from the list
    depth_buffer = depth_buffer[0]
    # We obtain the rgb image from the kinect
    _, _, rgb_image = coppeliasim_client.read_image_from_sensors([handle_kinect_rgb], color_mode='rgb')
    # We extract the numpy matrix out of the list
    rgb_image = rgb_image[0]
    # We return the depth buffer and the rgb image
    return rgb_image, depth_buffer


# We use this function to mark the contour colors on the rgb image
def mark_contour_colors_bgr(bgr_image, contours, contour_center_list, contour_color_list,
                            pickup_contour_center, pickup_orientation):
    # We iterate through the contours
    for j in range(len(contours)):
        # We check what color should be marked on the image
        if contour_color_list[j] == 0:
            # We store the rgb components for red
            [red, green, blue] = [1, 0, 0]
        elif contour_color_list[j] == 1:
            # We store the rgb components for green
            [red, green, blue] = [0, 1, 0]
        elif contour_color_list[j] == 2:
            # We store the rgb components for blue
            [red, green, blue] = [0, 0, 1]
        elif contour_color_list[j] == 3:
            # We store the rgb components for yellow
            [red, green, blue] = [1, 1, 0]
        elif contour_color_list[j] == 4:
            # We store the rgb components for pink
            [red, green, blue] = [1, 0, 1]
        else:
            # We store the rgb components for turquoise
            [red, green, blue] = [0, 1, 1]
        # We obtain the length of that particular contour
        current_contour = contours[j]
        # We iterate through each element of the contour
        for k in range(len(current_contour)):
            # We iterate through the following row offsets
            for m in range(-1, 2, 1):
                # We iterate through the following column offsets
                for n in range(-1, 2, 1):
                    # We mark that color of the contour within the rgb image
                    bgr_image[current_contour[k][0][1] + m, current_contour[k][0][0] + n] = [blue, green, red]
        # We obtain the contour center coordinates
        [contour_center_row, contour_center_column] = [int(contour_center_list[j][0]), int(contour_center_list[j][1])]
        # We iterate through the following row offsets
        for m in range(-1, 2, 1):
            # We iterate through the following column offsets
            for n in range(-1, 2, 1):
                # We mark the center of the contour within the rgb image as orange
                bgr_image[contour_center_row + m, contour_center_column + n] = [0, 0.65, 1]
    # We specify the distances needed to draw the lines
    distance_1 = 10
    distance_2 = 35
    # We iterate through both the combinations
    for j in [1, -1]:
        # We obtain coordinates to draw the line on one side of the pickup center point
        side_1_row = int(pickup_contour_center[0] + j * distance_1 * cos(pickup_orientation))
        side_1_column = int(pickup_contour_center[1] + j * distance_1 * sin(pickup_orientation))
        side_2_row = int(pickup_contour_center[0] + j * distance_2 * cos(pickup_orientation))
        side_2_column = int(pickup_contour_center[1] + j * distance_2 * sin(pickup_orientation))
        # Indices of pixels (rr, cc) and intensity values (val)
        row_index_list, column_index_list, value_list = line_aa(side_1_row, side_1_column, side_2_row, side_2_column)
        # We update the bgr image
        bgr_image[row_index_list, column_index_list] = [1, 1, 1]
    # We return the modified rgb image
    return bgr_image


# This function helps evaluate the color of the contours
def find_contour_colors(rgb_image, contour_center_list):
    # We record the number of contours
    contour_count = len(contour_center_list)
    # This variable records the color of the contour
    contour_colors = [0] * contour_count
    # We iterate through the contours in the list
    for j in range(contour_count):
        # We increment the color count for each color identified
        contour_colors[j] = find_pixel_color(rgb_image[int(contour_center_list[j][0]), int(contour_center_list[j][1])])
    # We return the index having maximum counts as the color group of the
    return contour_colors


# This function checks the color group of the rgb pixel provided
def find_pixel_color(pixel_color, color_separation_factor=5):
    # We check if the color category is red
    if pixel_color[0] > color_separation_factor * max(pixel_color[1], pixel_color[2]):
        # We return it as red
        return 0
    # We check if the color category is green
    elif pixel_color[1] > color_separation_factor * max(pixel_color[0], pixel_color[2]):
        # We return it as green
        return 1
    # We check if the color category is blue
    elif pixel_color[2] > color_separation_factor * max(pixel_color[0], pixel_color[1]):
        # We return it as blue
        return 2
    # We check if the color category is yellow
    elif color_separation_factor * pixel_color[2] < min(pixel_color[0], pixel_color[1]):
        # We return it as yellow
        return 3
    # We check if the color category is pink
    elif color_separation_factor * pixel_color[1] < min(pixel_color[0], pixel_color[2]):
        # We return it as pink
        return 4
    # We check if the color category is something else (turquoise)
    else:
        # We return it as turquoise
        return 5


# This function converts a rgb image to 3 layered grayscale image
def convert_rgb_to_grayscale_3_layered(rgb_image):
    # We obtain the global variables
    global rgb_resolution
    # We create a grayscale image which we will later modify
    grayscale_3_layered = np.zeros([rgb_resolution[0], rgb_resolution[1], 3], dtype=np.uint8)
    # we iterate through the row elements
    for j in range(rgb_resolution[0]):
        # We iterate through the column elements
        for k in range(rgb_resolution[1]):
            # We obtain the value to write at this location
            gray_value = np.uint8(sum(rgb_image[j, k])/3.0)
            # We write the grayscale value at that location
            grayscale_3_layered[j, k] = [gray_value, gray_value, gray_value]
    # We return the grayscale matrix
    return grayscale_3_layered

# ----- Functions that facilitate kuka arm functionality -----


# This function makes the kuka arm go to the conveyor belt
def kuka_arm_move_to_destination(destination='home'):
    # We access the latest kuka joint angles
    global multiple_joint_handles, latest_multiple_joint_angles
    # We check if we are to move to the rasp position, trash, home, or to any one of the conveyor belts
    if destination == 'grasping':
        # We set the kuka joint angles to match the grasping position
        set_kuka_joint_angles = [0, 30 * np.pi/180, 0, -85 * np.pi/180, 0, 65 * np.pi/180, 0]
    elif destination == 'trash':
        # We set the kuka joint angles to match the trash bin position
        set_kuka_joint_angles = [-40 * np.pi / 180, 90 * np.pi/180, 0, -0 * np.pi/180, 0, 90 * np.pi/180, 0]
    elif destination == 'conveyor_1':
        # We set the kuka joint angles to match the conveyor belt 1 position
        set_kuka_joint_angles = [-90 * np.pi / 180, 45 * np.pi/180, 0, -85 * np.pi/180, 0, 50 * np.pi/180, 0]
    elif destination == 'conveyor_2':
        # We set the kuka joint angles to match the conveyor belt 2 position
        set_kuka_joint_angles = [0, -45 * np.pi/180, 0, 85 * np.pi/180, 0, -50 * np.pi/180, 0]
    elif destination == 'conveyor_3':
        # We set the kuka joint angles to match the conveyor belt 3 position
        set_kuka_joint_angles = [90 * np.pi / 180, 45 * np.pi/180, 0, -85 * np.pi/180, 0, 50 * np.pi/180, 0]
    else:
        # We set the kuka joint angles to match the home position
        set_kuka_joint_angles = [0, 0, 0, 0, 0, 0, 0]
    # We set the kuka arm to go to to the required destination
    coppeliasim_client.control_joint_property(multiple_joint_handles[0:7], set_kuka_joint_angles, 'target_position')
    # We sleep for some time
    time.sleep(0.1)
    # We run a while loop until we have reached the desired coordinates
    while not check_arm_configuration_equality(set_kuka_joint_angles, latest_multiple_joint_angles[0:7]):
        # We sleep for some time
        time.sleep(0.1)


# This function checks whether two kuka arm configurations are approximately the same
def check_arm_configuration_equality(configuration_1_joint_angles, configuration_2_joint_angles):
    # We obtain the length of the first and second arm configurations
    configuration_1_length, configuration_2_length = len(configuration_1_joint_angles), \
                                                     len(configuration_2_joint_angles)
    # We check whether the configuration lengths are matching
    if configuration_1_length != configuration_2_length:
        # We return that the configurations do not match
        return False
    # This variable indicates whether the configurations approximately match
    configuration_match = True
    # We iterate through the entire configuration length
    for j in range(configuration_1_length):
        # We check if either configuration angle is deviating by more than 1 degree
        if configuration_1_joint_angles[j] > configuration_2_joint_angles[j] + 1 * np.pi/180 or \
           configuration_1_joint_angles[j] < configuration_2_joint_angles[j] - 1 * np.pi/180:
            # We indicate that the configurations do not match
            configuration_match = False
            # We break out of the for loop as further iterations don't affect the findings
            break
    # We return the obtained configuration
    return configuration_match


# This function opens and closes the kuka arm gripper
def kuka_arm_gripper_control(action='close'):
    # We access the latest kuka joint angles
    global multiple_joint_handles
    # We check if the action to be taken is to close the gripper
    if action == 'close':
        # We set the command to close the gripper
        coppeliasim_client.control_gripper(multiple_joint_handles[7], multiple_joint_handles[8], 'close', 3.0)
    elif action == 'open':
        # We set the command to close the gripper
        coppeliasim_client.control_gripper(multiple_joint_handles[7], multiple_joint_handles[8], 'open', 3.0)


# This function uses the proximity sensor to detect if we have gripped an object
def check_if_object_in_gripper():
    # We access the latest kuka joint angles
    global latest_proximity_state_list, latest_proximity_point_list
    # We check if we are detecting a point in the gripper
    if latest_proximity_state_list[1] and latest_proximity_point_list[1] < 0.03:
        # We return that we have detected a point in front of the gripper proximity sensor
        return True
    # We have not detected anything 5cm away from the gripper center
    else:
        # We return that no object is in the gripper
        return False


# This function handles the kuka arm operations
def kuka_arm_pickup_and_drop():
    # We access the global variables that represent the kuka arm parameters
    global gripper_height, media_flange_height, kuka_arm_length_dict, kuka_joint_range_dict, \
        arm_pickup_point_and_object_color, latest_multiple_joint_angles, multiple_joint_handles, \
        latest_simulation_time, latest_proximity_state_list, latest_proximity_point_list, robot_pickup_ready
    # We check if we have an object ready to pickup and drop
    if robot_pickup_ready and arm_pickup_point_and_object_color is not None:
        # We set the kuka arm to move to the grasping position above conveyor belt 4
        kuka_arm_move_to_destination(destination='grasping')
        # We find the angle of kuka joint 1 to hover over the object placed on conveyor 4
        kuka_joint_1_angle = atan(arm_pickup_point_and_object_color[1]/arm_pickup_point_and_object_color[0])
        # We find the effective gripper angle due to the offset created by kuka joint angle one
        kuka_gripper_angle = kuka_joint_1_angle - arm_pickup_point_and_object_color[3]
        # We obtain the horizontal distance to the object to be captured
        horizontal_distance = (arm_pickup_point_and_object_color[0]**2 + arm_pickup_point_and_object_color[1]**2)**0.5
        # We create an inverse kinematics model for kuka arm segments 2 and 3
        kuka_arm_segment = tinyik.Actuator(['z', [kuka_arm_length_dict['kuka_length_2'], 0., 0.],
                                            'z', [kuka_arm_length_dict['kuka_length_3'], 0., 0.]])
        # We specify the joint handles we are controlling
        pick_up_joint_handles = multiple_joint_handles[0:2] + [multiple_joint_handles[3]] + \
            multiple_joint_handles[5:7]
        # We set the vertical gripping offset to start the pick up procedure
        vertical_gripping_offset = 0.1
        # We set the vertical gripping increments to move the robotic arm
        vertical_gripping_increment = 0.005
        # The adjustment height for gripping
        vertical_adjustment_height = 0.055
        # We obtain the vertical distance at which we attempt to grip the object
        vertical_distance = - kuka_arm_length_dict['kuka_length_1'] + kuka_arm_length_dict['kuka_length_4'] + \
            media_flange_height + gripper_height + arm_pickup_point_and_object_color[2] + vertical_adjustment_height
        # This flag indicates whether any gripper blocking was detected
        gripper_block_flag = False
        # We run a while loop until we have the block between the gripper
        while not gripper_block_flag and vertical_gripping_offset > 0 and not \
                (latest_proximity_state_list[1] and latest_proximity_point_list[1] < 0.02):
            # Inverse kinematics - Sets a position of the end-effector to calculate the joint angles
            kuka_arm_segment.ee = [horizontal_distance, vertical_distance + vertical_gripping_offset, 0.0]
            # Inverse kinematics - Extracts the calculated joint angles
            kuka_joint_2_angle = np.pi/2 - kuka_arm_segment.angles[0]
            kuka_joint_4_angle = kuka_arm_segment.angles[1]
            # We check if the kuka joint angle 2 is greater than 90 degrees
            if kuka_joint_4_angle > 0:
                # We obtain the angle of the line between two vertical sections of the arm
                intermediate_angle = np.pi/2 - atan((vertical_distance + vertical_gripping_offset) / horizontal_distance)
                # We obtain the angle to mirror kuka arm links 2 and 3
                mirror_angle = intermediate_angle - kuka_joint_2_angle
                # We use the mirror angle to adjust the angle of joint 2
                kuka_joint_2_angle += 2 * mirror_angle
                # We mirror the value of joint 4 as well
                kuka_joint_4_angle = - kuka_joint_4_angle
            # We know that joint angle 6 is 180 degrees minus other two angles
            kuka_joint_6_angle = np.pi - kuka_joint_2_angle + kuka_joint_4_angle
            # We specify the joint angles we are controlling for the pickup operation
            pick_up_joint_angles = [kuka_joint_1_angle, kuka_joint_2_angle, kuka_joint_4_angle,
                                    kuka_joint_6_angle, kuka_gripper_angle]
            # We set the kuka arm to go to to the required destination
            coppeliasim_client.control_joint_property(pick_up_joint_handles, pick_up_joint_angles, 'target_position')
            # We note the pickup increment start time
            pickup_increment_start_time = deepcopy(latest_simulation_time)
            # We run a while loop until we have reached the desired coordinates
            while not check_arm_configuration_equality(pick_up_joint_angles,
                                                       [latest_multiple_joint_angles[j] for j in [0, 1, 3, 5, 6]]):
                # We sleep for 0.05 seconds before checking again
                time.sleep(0.05)
                # We check too much time has passed, indicating blockage
                if abs(latest_simulation_time - pickup_increment_start_time) > 3:
                    # We raise the gripper block flag
                    gripper_block_flag = True
                    # We break out of the while loop
                    break
            # We update the vertical offset by the vertical gripping increment
            vertical_gripping_offset -= vertical_gripping_increment
        # We check if we were not blocked during the operation
        if not gripper_block_flag:
            # We must now close the gripper
            kuka_arm_gripper_control(action='close')
            # We note the time at which we start closing the gripper
            gripper_close_start_time = deepcopy(latest_simulation_time)
            # We run a while loop until we feel the gripper is sufficiently closed
            while abs(latest_simulation_time - gripper_close_start_time) < 1.5:
                # We sleep for 0.1 seconds before checking again
                time.sleep(0.1)
            while vertical_gripping_offset <= 0.1:
                # Inverse kinematics - Sets a position of the end-effector to calculate the joint angles
                kuka_arm_segment.ee = [horizontal_distance, vertical_distance + vertical_gripping_offset, 0.0]
                # Inverse kinematics - Extracts the calculated joint angles
                kuka_joint_2_angle = np.pi/2 - kuka_arm_segment.angles[0]
                kuka_joint_4_angle = kuka_arm_segment.angles[1]
                # We check if the kuka joint angle 2 is greater than 90 degrees
                if kuka_joint_4_angle > 0:
                    # We obtain the angle of the line between two vertical sections of the arm
                    intermediate_angle = np.pi/2 - atan((vertical_distance + vertical_gripping_offset) / horizontal_distance)
                    # We obtain the angle to mirror kuka arm links 2 and 3
                    mirror_angle = intermediate_angle - kuka_joint_2_angle
                    # We use the mirror angle to adjust the angle of joint 2
                    kuka_joint_2_angle += 2 * mirror_angle
                    # We mirror the value of joint 4 as well
                    kuka_joint_4_angle = - kuka_joint_4_angle
                # We know that joint angle 6 is 180 degrees minus other two angles
                kuka_joint_6_angle = np.pi - kuka_joint_2_angle + kuka_joint_4_angle
                # We specify the joint angles we are controlling for the pickup operation
                pick_up_joint_angles = [kuka_joint_1_angle, kuka_joint_2_angle, kuka_joint_4_angle,
                                        kuka_joint_6_angle, kuka_gripper_angle]
                # We set the kuka arm to go to to the required destination
                coppeliasim_client.control_joint_property(pick_up_joint_handles, pick_up_joint_angles, 'target_position')
                # We run a while loop until we have reached the desired coordinates
                while not check_arm_configuration_equality(pick_up_joint_angles,
                                                           [latest_multiple_joint_angles[j] for j in [0, 1, 3, 5, 6]]):
                    # We sleep for 0.05 seconds before checking again
                    time.sleep(0.05)
                # We update the vertical offset by the vertical gripping increment
                vertical_gripping_offset += vertical_gripping_increment
        # Regardless of whether blocked or unblocked we must raise the gripper back to the grasping position
        kuka_arm_move_to_destination(destination='grasping')
        # We must now check if there still is an item in the gripper head
        # ISSUE - if latest_proximity_state_list[1]: (Some bug is preventing reading of this proximity sensor)
        if True:
            # We check if the object to be picked up is red
            if arm_pickup_point_and_object_color[4] == 0:
                # We move the object over conveyor belt 1
                kuka_arm_move_to_destination(destination='conveyor_1')
            # We check if the object to be picked up is green
            elif arm_pickup_point_and_object_color[4] == 1:
                # We move the object over conveyor belt 2
                kuka_arm_move_to_destination(destination='conveyor_2')
            # We check if the object to be picked up is blue
            elif arm_pickup_point_and_object_color[4] == 2:
                # We move the object over conveyor belt 3
                kuka_arm_move_to_destination(destination='conveyor_3')
            # The object we picked is some other color
            else:
                # We move the object over the trash
                kuka_arm_move_to_destination(destination='trash')
            # We note the time at which we arrived at the drop off point
            drop_off_arrival_time = deepcopy(latest_simulation_time)
            # We run a while loop until we feel the gripper is sufficiently closed
            while abs(latest_simulation_time - drop_off_arrival_time) < 1:
                # We sleep for 0.1 seconds before checking again
                time.sleep(0.1)
            # We open the gripper to release the object
            kuka_arm_gripper_control(action='open')
            # We wait until the object falls out of the gripper
            while check_if_object_in_gripper():
                # We sleep for 0.1 seconds
                time.sleep(0.1)
            # We note the time at which we arrived at the drop off point
            drop_off_arrival_time = deepcopy(latest_simulation_time)
            # We run a while loop until we feel the gripper is sufficiently closed
            while abs(latest_simulation_time - drop_off_arrival_time) < 1:
                # We sleep for 0.1 seconds before checking again
                time.sleep(0.1)
            # We check if the object picked up was meant for a conveyor belt
            if arm_pickup_point_and_object_color[4] <= 2:
                # We specify that we have loaded the conveyor belt(1-3) with an object
                conveyor_loaded_status[arm_pickup_point_and_object_color[4]] = True
            # Now that we have dropped the item, we return to the grasping position
            kuka_arm_move_to_destination(destination='grasping')
        # We probably dropped the object back onto conveyor belt 4
        else:
            # In this case we must open the gripper anyway for the next round
            kuka_arm_gripper_control(action='open')
            # We note the time at which we start opening the gripper
            gripper_open_start_time = deepcopy(latest_simulation_time)
            # We run a while loop until we feel the all fallen blocks have settled (if any in a rare case)
            while abs(latest_simulation_time - gripper_open_start_time) < 1:
                # We sleep for 0.1 seconds before checking again
                time.sleep(0.1)
        # Regardless of whether we were successful or not we say the robot is no longer ready for pickup
        robot_pickup_ready = False

# ----- The main loop where the program is executed -----

# We run the while loop as long as any object is on a conveyor belt. Program ends when all are in sorting boxes
while any(conveyor_loaded_status):
    # We update the simulation time
    latest_simulation_time = coppeliasim_client.get_simulation_time()
    # We obtain the latest coordinates of all the joints of the kuka arm and the pusher head
    _, latest_multiple_joint_angles = coppeliasim_client.retrieve_joint_property(multiple_joint_handles,
                                                                                 selected_property='position')
    # We check if we have set the image update flag as false in a thread
    if not image_update_flag:
        # We obtain the latest rgb image and depth buffer from the kinect sensor
        latest_rgb_image, latest_depth_buffer = capture_rgb_image_and_depth_buffer()
        # We update the image update flag
        image_update_flag = True
    else:
        # We obtain the latest rgb image and depth buffer from the kinect sensor
        latest_rgb_image, latest_depth_buffer = capture_rgb_image_and_depth_buffer()
    # We obtain the latest proximity sensor readings
    _, latest_proximity_state_list, latest_proximity_point_list = \
        coppeliasim_client.read_proximity_sensors(proximity_sensor_handle_list)
    # We iterate through the 1-3 conveyor belts and process them
    for i in range(3):
        # We check if the conveyor belt is loaded and is available for moving the objects to the collection bins
        if conveyor_loaded_status[i] and (conveyor_threads[i] is None or conveyor_threads[i].done()):
            # We create a thread to handle the chosen conveyor belt movement
            conveyor_threads[i] = conveyor_thread_pool.submit(move_onto_collection_bin, i+1)
    # We check if the conveyor 6 has to dump an object onto conveyor 5
    if not conveyor_loaded_status[4] and conveyor_loaded_status[5] and (conveyor_threads[3] is None or
                                                                        conveyor_threads[3].done()):
        # We load an object onto conveyor belt 5
        conveyor_threads[3] = conveyor_thread_pool.submit(load_onto_conveyor_5)
    # We check if the conveyor 5 has to dump an object onto conveyor 4
    if not conveyor_loaded_status[3] and conveyor_loaded_status[4] and (conveyor_threads[4] is None or
                                                                        conveyor_threads[4].done()):
        # We load an object onto conveyor belt 4
        conveyor_threads[4] = conveyor_thread_pool.submit(load_onto_conveyor_4)
    # We initialize conveyor 7 thread just once
    if conveyor_loaded_status[6] and (conveyor_threads[5] is None or conveyor_threads[5].done()):
        # We start loading objects from conveyor 7  onto conveyor 6
        conveyor_threads[5] = conveyor_thread_pool.submit(move_conveyor_7_till_end)
    # Now that we have finished handling all other conveyors in this round, we will focus on conveyor 4
    if conveyor_loaded_status[3] and not robot_pickup_ready and (conveyor_threads[6] is None or
                                                                 conveyor_threads[6].done()):
        # We move conveyor belt 4 until an object is identified by kinect to be picked up by the arm
        conveyor_threads[6] = conveyor_thread_pool.submit(move_conveyor_4_until_object_identified)
    # Now that we have an object ready for pickup by the robotic arm we run the kuka arm thread
    if robot_pickup_ready and (conveyor_threads[7] is None or conveyor_threads[7].done()):
        # We move the kuka arm to pickup the object and drop it off at the required location
        conveyor_threads[7] = conveyor_thread_pool.submit(kuka_arm_pickup_and_drop)
    # We check if we are not in the function that evaluates contours using kinect streams
    if not image_buffer_lock:
        # We create an image buffer variable to help display images in real time
        image_buffer = cv2.cvtColor(cv2.cvtColor(latest_rgb_image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
    # We display the image obtained by the depth camera
    cv2.imshow('Object Recognition Results', image_buffer)
    # This will display a frame for 1 ms and since the previous function is called multiple times, updates it
    cv2.waitKey(1)
    # We sleep at the end of each round for 0.05 seconds
    time.sleep(0.05)
# We reset the Kuka arm as the program has ended
kuka_arm_move_to_destination(destination='home')

# ----- Closing connection to the coppeliasim server -----

# Now close the connection to CoppeliaSim
coppeliasim_client.disconnect_from_server()
print('Mission Accomplished')

# ----- End of Code -----
