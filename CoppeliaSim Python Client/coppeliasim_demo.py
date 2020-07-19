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
import time
import sys
import numpy as np

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
# We inform the user that we have connected succesfully to the coppeliasim server
print('Connected to the coppeliasim server')

# ----- Getting the handles of all actuators in the Kuka arm of the coppeliasim scene -----

# Example 1: moving the pusher head and retreiving its position
object_error_list, object_handle_list = coppeliasim_client.get_object_handle(['pusher_head_joint'])
print(coppeliasim_client.control_joint_property(object_handle_list, [0.2], selected_property='target_position'))
time.sleep(1)
coppeliasim_client.retrieve_joint_property(object_handle_list, 'position')

# Example 2: reading kinect RGB and depth
object_error_list, object_handle_list = coppeliasim_client.get_object_handle(['kinect_rgb', 'kinect_depth'])
error_list, resolutions, images = coppeliasim_client.read_image_from_sensors([object_handle_list[0]], color_mode='rgb')
error_list, resolutions, buffer = coppeliasim_client.read_depth_buffer_from_sensors([object_handle_list[1]])

# Example 3: controlling gripper
object_error_list, object_handle_list = coppeliasim_client.get_object_handle(['ROBOTIQ_85_active1',
                                                                              'ROBOTIQ_85_active2'])
coppeliasim_client.control_gripper(object_handle_list[0], object_handle_list[1], 'close', 2.0)
time.sleep(2)
coppeliasim_client.control_gripper(object_handle_list[0], object_handle_list[1], 'open', 1.0)

# Example 4: controlling conveyors
coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_5'], [0.1])
coppeliasim_client.set_conveyor_speed(['conveyorBeltVelocity_4'], [0.1])
coppeliasim_client.set_conveyor_distance_and_speed(['conveyorBeltVelocity_1'], [1], [0.05])

# Example 5: Moving kuka joints
object_error_list, object_handle_list = coppeliasim_client.get_object_handle(['kuka_joint_1', 'kuka_joint_4',
                                                                              'kuka_joint_2'])
coppeliasim_client.control_joint_property(object_handle_list, [0.5, 0.8, 0.2], selected_property='target_position')



# ----- Closing connection to the coppeliasim server -----

# Now close the connection to CoppeliaSim
coppeliasim_client.disconnect_from_server()

# ----- End of Code -----
