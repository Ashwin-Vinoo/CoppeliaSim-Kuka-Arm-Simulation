# ---------------- COPPELIASIM CLIENT CLASS ---------------

# This code specifies the coppeliasim client class which may be used to interface to the coppeliasim server
# and initiate control of the different actuators and sensors
#
# Created by Ashwin Vinoo - 5/8/2020

# Please refer https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm

# ----- Importing all the necessary modules -----
import sim
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from warnings import warn

# ----- Coppeliasim Client Class Definition -----


# We define the client class below:
class CoppeliaSimClient:

    # We override the class constructor
    def __init__(self):
        # Helps store the client id allocated by the coppeliasim server
        self.client_id = -1
        # We want one parallel process to be running at the most to handle conveyor timings
        self.conveyor_timer_pool = ThreadPoolExecutor(max_workers=1)
        # We use this variable to hold the current active thread
        self.current_thread = None
        # Helps store the end time of several conveyor movements
        self.conveyor_belt_velocity_timers = {}
        # This function helps identify the first time several functions with streaming recommendations have to be run
        self.first_time_list = [True, True, True, True, True, True, True, True, True]

    # ----- Functions to connect and disconnect from coppeliasim server and get the simulation time -----

    # This function helps connect to the coppeliasim server. input: port number, output: boolean indicating success/fail
    def connect_to_server(self, port=19999, timeout_ms=5000):
        # We close all previous connections that may still be open
        sim.simxFinish(-1)
        # We connect to coppeliasim by launching the coppeliasim client (connection address or local host, port number,
        # waitUntilConnected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs)
        self.client_id = sim.simxStart('127.0.0.1', port, True, True, timeout_ms, 1)
        # We check if the client ID is returned as negative (failed to connect)
        if self.client_id == -1:
            # We inform the user that we have not connected successfully to the coppeliasim server
            return False
        # We inform the user that we have connected successfully to the coppeliasim server
        return True

    # This function should be called at the end to terminate connection to the coppeliasim server
    def disconnect_from_server(self):
        # We wait for all the conveyor timers to expire
        while self.current_thread is not None:
            # We skip the execution of this loop
            pass
        # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive
        sim.simxGetPingTime(self.client_id)
        # Now close the connection to CoppeliaSim
        sim.simxFinish(self.client_id)
        # Helps store the client id allocated by the coppeliasim server
        self.client_id = -1

    # We use this function to get the simulation time (Deprecated)
    def get_simulation_time_2(self):
        # We run a function in a child script in coppeliasim that returns the simulation time
        error_code, _, result_float, _, _ = \
            sim.simxCallScriptFunction(self.client_id, 'kuka_assembly', sim.sim_scripttype_childscript,
                                       'python_get_sim_time', [], [], '', bytearray(), sim.simx_opmode_blocking)
        # We return the simulation time
        return result_float[0]

    # We use this function to get the simulation time
    def get_simulation_time(self):
        # We get the error code and floating signal back from the server
        if self.first_time_list[0]:
            _, simulation_time = sim.simxGetFloatSignal(self.client_id, 'simulation_time', sim.simx_opmode_streaming)
            self.first_time_list[0] = False
        else:
            _, simulation_time = sim.simxGetFloatSignal(self.client_id, 'simulation_time', sim.simx_opmode_buffer)
        # We return the simulation time
        return simulation_time

    # We use this to wait for a time period to traverse in the simulation (equivalent of time.sleep())
    def simulation_sleep(self, time_to_sleep):
        # We get the current time in the simulator
        start_time = self.get_simulation_time()
        # We iterate through a for loop
        while abs(self.get_simulation_time() - start_time) < time_to_sleep:
            # We sleep for 0.1 seconds
            sleep(0.1)

    # ----- Functions to obtain handles and coordinates for every object (joint, sensor or shape) -----

    # This function returns an error code and object handle list for the passed object name list (error code 0 is good)
    def get_object_handle(self, object_name_list):
        # We obtain the length of the object name list
        object_list_length = len(object_name_list)
        # We create an empty list to hold the object handles that will be returned from the server
        object_handle_list = [0]*object_list_length
        # We create an empty list to hold the object error code that will be returned from the server
        object_error_list = [0]*object_list_length
        # We iterate through the objects of the handle list
        for i in range(object_list_length):
            # We obtain the error code and handle for each of the specified objects
            object_error_list[i], object_handle_list[i] = sim.simxGetObjectHandle(self.client_id, object_name_list[i],
                                                                                  sim.simx_opmode_blocking)
        # We return the error and object handle list
        return object_error_list, object_handle_list

    # This function retrieves the coordinates of the list of objects with respect to a reference handle (-1 is with
    # respect to world frame)
    def get_object_coordinates(self, object_handle_list, relative_to_object_handle=-1):
        # We obtain the length of the object handle list
        object_list_length = len(object_handle_list)
        # We create an empty list to hold the object positions that will be returned from the server
        object_position_list = [0]*object_list_length
        # We create an empty list to hold the object orientations that will be returned from the server
        object_orientation_list = [0]*object_list_length
        # We create an empty list to hold the object error code that will be returned from the server
        object_error_list = [0]*object_list_length
        # We iterate through the objects of the handle list
        for i in range(object_list_length):
            # We obtain the error code and handle for each of the specified objects
            object_error_list[i], object_position_list[i] = \
                sim.simxGetObjectPosition(self.client_id, object_handle_list[i], relative_to_object_handle,
                                          sim.simx_opmode_blocking)
            # We obtain the error code and handle for each of the specified objects
            _, object_orientation_list[i] = \
                sim.simxGetObjectOrientation(self.client_id, object_handle_list[i], relative_to_object_handle,
                                             sim.simx_opmode_blocking)
        # We return the error and the object handle list
        return object_error_list, object_position_list, object_orientation_list

    # ----- Functions to control and retrieve the properties of all joints -----


    # This function helps control the selected property of each joint in the list
    def control_joint_property(self, joint_handle_list, joint_property_list, selected_property='target_position'):
        # We obtain the length of the joint name list
        joint_list_length = len(joint_handle_list)
        # We create an empty list to hold the joint error code that will be returned from the server
        joint_error_list = [0]*joint_list_length
        # We identify the function that matches the selected property
        if selected_property == 'target_position':
            function_used = sim.simxSetJointTargetPosition
        elif selected_property == 'position':
            function_used = sim.simxSetJointPosition
        elif selected_property == 'target_velocity':
            function_used = sim.simxSetJointTargetVelocity
        elif selected_property == 'joint_force':
            function_used = sim.simxSetJointForce
        else:
            warn("Invalid joint property selected. Returning empty lists")
            # Since we got an unknown joint property requested we will return nothing
            return []
        # We iterate through the joint handles
        for i in range(joint_list_length):
            # We write the joint target positions for each joint handle
            joint_error_list[i] = function_used(self.client_id, joint_handle_list[i], joint_property_list[i],
                                                sim.simx_opmode_oneshot)
        # We return the error returned for each joint of the handle list
        return joint_error_list


    # This function helps retrieve the selected property of each joint in the list
    def retrieve_joint_property(self, joint_handle_list, selected_property='position'):
        # We obtain the length of the joint name list
        joint_list_length = len(joint_handle_list)
        # We create an empty list to hold the joint error code that will be returned from the server
        joint_error_list = [0]*joint_list_length
        # We create an empty list to hold the property values returned from the server
        joint_property_list = [0]*joint_list_length
        # We initialize this variable with False. It denotes whether we are using the property function the first time
        first_time = False
        # We identify the function that matches the selected property
        if selected_property == 'position':
            # We identify the function to be used
            function_used = sim.simxGetJointPosition
            # We check if this is the first time using this function
            if self.first_time_list[1]:
                # We mark that first time is true
                first_time = True
                # We mark it as false in the list
                self.first_time_list[1] = False
        elif selected_property == 'joint_matrix':
            # We identify the function to be used
            function_used = sim.simxGetJointMatrix
            # We check if this is the first time using this function
            if self.first_time_list[2]:
                # We mark that first time is true
                first_time = True
                # We mark it as false in the list
                self.first_time_list[2] = False
        elif selected_property == 'joint_force':
            # We identify the function to be used
            function_used = sim.simxGetJointForce
            # We check if this is the first time using this function
            if self.first_time_list[3]:
                # We mark that first time is true
                first_time = True
                # We mark it as false in the list
                self.first_time_list[3] = False
        else:
            warn("Invalid joint property selected. Returning empty lists")
            # Since we got an unknown joint property requested we will return with error
            return [], []
        # We iterate through the joint handles
        for i in range(joint_list_length):
            # We check if this is the first time running the specified function
            if first_time:
                # We write the joint target positions for each joint handle in streaming mode
                joint_error_list[i], joint_property_list[i] = function_used(self.client_id, joint_handle_list[i],
                                                                            sim.simx_opmode_streaming)
                # We sleep for 0.5 seconds
                sleep(0.5)
                # We write the joint target positions for each joint handle in buffer mode
                joint_error_list[i], joint_property_list[i] = function_used(self.client_id, joint_handle_list[i],
                                                                            sim.simx_opmode_buffer)
            # It is not the first time we have used this particular function
            else:
                # We write the joint target positions for each joint handle in buffer mode
                joint_error_list[i], joint_property_list[i] = function_used(self.client_id, joint_handle_list[i],
                                                                            sim.simx_opmode_buffer)
        # We return the error returned for each joint of the handle list and the joint property list
        return joint_error_list, joint_property_list

    # ----- Functions to control the conveyor belts -----


    # This function helps control the speed at which the conveyor is moving (runs forever at specified speed)
    def set_conveyor_speed(self, conveyor_property_list, conveyor_speed_list):
        # We iterate through the conveyor name list
        for i in range(len(conveyor_property_list)):
            # We set the conveyor speed (m/s)
            sim.simxSetFloatSignal(self.client_id, conveyor_property_list[i],
                                   conveyor_speed_list[i], sim.simx_opmode_oneshot)

    # We use this function to check the speed of conveyor 4 alone
    def get_conveyor_4_speed(self):
        # We get the error code and floating signal back from the server
        if self.first_time_list[7]:
            _, conveyor_4_speed = sim.simxGetFloatSignal(self.client_id,
                                                         'active_conveyorBeltVelocity_4', sim.simx_opmode_streaming)
            self.first_time_list[7] = False
        else:
            _, conveyor_4_speed = sim.simxGetFloatSignal(self.client_id,
                                                         'active_conveyorBeltVelocity_4', sim.simx_opmode_buffer)
        # We return the simulation time
        return conveyor_4_speed

    # This function helps the conveyor belts traverse a certain distance at the specified speed
    def set_conveyor_distance_and_speed(self, conveyor_property_list, conveyor_distance_list, conveyor_speed_list):
        # We set the conveyor belts to move at the speeds specified
        self.set_conveyor_speed(conveyor_property_list, conveyor_speed_list)
        # We iterate through the conveyor name list
        for i in range(len(conveyor_property_list)):
            # We set the conveyor speed (m/s)
            sim.simxSetFloatSignal(self.client_id, conveyor_property_list[i],
                                   conveyor_speed_list[i], sim.simx_opmode_oneshot)
            # We know that time to run the conveyor(s) = distance(m)/speed(m/s)
            time_to_finish_run = self.get_simulation_time() + conveyor_distance_list[i]/conveyor_speed_list[i]
            # We update the conveyor velocity dictionary
            self.conveyor_belt_velocity_timers[conveyor_property_list[i]] = time_to_finish_run
            # We check if a conveyor thread is not running
            if self.current_thread is None:
                # We run a parallel thread that monitors conveyor belt movements
                self.current_thread = self.conveyor_timer_pool.submit(self.conveyor_thread_handler)

    # This function is run as a parallel process that check conveyor timers continuously
    def conveyor_thread_handler(self):
        # We check if the dictionary that holds conveyor timings is empty
        while len(self.conveyor_belt_velocity_timers) > 0:
            # We sleep for 0.1 seconds between each thread execution round
            sleep(0.1)
            # We get the current simulation time
            current_sim_time = self.get_simulation_time()
            # We iterate through the number of terms in the dictionary
            for i in range(len(self.conveyor_belt_velocity_timers)):
                # We check whether the times stored in the dictionary have passed
                if self.conveyor_belt_velocity_timers[list(self.conveyor_belt_velocity_timers)[i]] < current_sim_time:
                    # We set the conveyor velocity to zero
                    self.set_conveyor_speed([list(self.conveyor_belt_velocity_timers)[i]], [0])
                    # We remove that conveyor velocity from the dictionary
                    self.conveyor_belt_velocity_timers.pop(list(self.conveyor_belt_velocity_timers)[i])
        # We specify that there is no thread currently working
        self.current_thread = None

    # ----- Function to open/close the gripper -----

    # This function handles gripper opening and closing with the speed multiplier provided
    def control_gripper(self, gripper_joint_handle_1, gripper_joint_handle_2, action_to_take,
                        action_speed_multiplier=1.0):
        # We identify the position of both the gripper joints
        error_code_1, position_1 = sim.simxGetJointPosition(self.client_id, gripper_joint_handle_1,
                                                            sim.simx_opmode_blocking)
        error_code_2, position_2 = sim.simxGetJointPosition(self.client_id, gripper_joint_handle_2,
                                                            sim.simx_opmode_blocking)
        # We report if any of the error codes are not matching
        if error_code_1 != 0 or error_code_2 != 0:
            # We warn the user that the handle(s) had an error
            warn("Gripper joint handle error detected")
        # We check which action is to be taken
        if action_to_take == 'close':
            if position_1 < position_2-0.008:
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_1, -0.01 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_2, -0.04 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
            else:
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_1, -0.04 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_2, -0.04 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
        elif action_to_take == 'open':
            if position_1 < position_2:
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_1, 0.04 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_2, 0.02 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
            else:
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_1, 0.02 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(self.client_id, gripper_joint_handle_2, 0.04 * action_speed_multiplier,
                                               sim.simx_opmode_oneshot)
        else:
            # We warn the user that the input was invalid for controlling the gripper
            warn("No action taken on gripper due to invalid inputs")

    # ----- Function to obtain readings from proximity sensors -----

    # This function helps retrieve readings from a list of proximity sensors
    def read_proximity_sensors(self, proximity_sensor_handle_list):
        # We obtain the length of the joint name list
        sensor_list_length = len(proximity_sensor_handle_list)
        # We create an empty list to hold the joint error code that will be returned from the server
        sensor_error_list = [0]*sensor_list_length
        # We create an empty list to hold the detection state of the proximity sensor
        detection_state_list = [False]*sensor_list_length
        # We create an empty list to hold the point detected by the proximity sensor with respect to the sensor frame
        detected_point_list = [[0, 0, 0]]*sensor_list_length
        # We iterate through the proximity sensor handle list
        for i in range(sensor_list_length):
            # We check if it is the first time running the function
            if self.first_time_list[6]:
                # We obtain the result from coppeliasim in streaming mode
                sensor_error_list[i], detection_state_list[i], detected_point_list[i], _, _ = \
                    sim.simxReadProximitySensor(self.client_id, proximity_sensor_handle_list[i],
                                                sim.simx_opmode_streaming)
                # We update the list and specify that we have run the function before
                self.first_time_list[6] = False
            # We have run this function before
            else:
                # We obtain the result from coppeliasim in buffer mode
                sensor_error_list[i], detection_state_list[i], detected_point_list[i], _, _ = \
                    sim.simxReadProximitySensor(self.client_id, proximity_sensor_handle_list[i],
                                                sim.simx_opmode_buffer)
            # We only care about the absolute distance
            detected_point_list[i] = (detected_point_list[i][0]**2 + detected_point_list[i][1]**2 +
                                      detected_point_list[i][2]**2)**0.5
        # We return the error returned for each joint of the handle list and the joint property list
        return sensor_error_list, detection_state_list, detected_point_list

    # ----- Functions to obtain readings from the kinect -----

    # This function helps retrieve images from camera sensors
    def read_image_from_sensors(self, image_sensor_handle_list, color_mode='rgb'):
        # We obtain the length of the image sensor handle list
        sensor_list_length = len(image_sensor_handle_list)
        # We create an empty list to hold the error code that will be returned for each image sensor
        sensor_error_list = [0]*sensor_list_length
        # We create an empty list to hold the resolutions of each sensor
        sensor_resolution_list = [[0, 0]]*sensor_list_length
        # We create an empty list to hold the images returned by each sensor
        sensor_image_list = [0]*sensor_list_length
        # We identify the requested color_mode
        if color_mode == 'rgb':
            color_mode = 0
        elif color_mode == 'grayscale':
            color_mode = 1
        else:
            warn("Invalid image color mode selected. Returning empty lists")
            # Since we got an unknown joint property requested we will return nothing
            return [], [], []
        # We iterate through the proximity sensor handle list
        for i in range(sensor_list_length):
            # We check if it is the first time running the function
            if self.first_time_list[4]:
                # We obtain the result from coppeliasim in streaming mode
                _, _, _ = sim.simxGetVisionSensorImage(
                    self.client_id, image_sensor_handle_list[i], color_mode, sim.simx_opmode_streaming)
                # We sleep for some time so that coppeliasim may setup the stream
                sleep(0.5)
                # We obtain the result from coppeliasim in buffer mode
                sensor_error_list[i], sensor_resolution_list[i], sensor_image = sim.simxGetVisionSensorImage(
                    self.client_id, image_sensor_handle_list[i], color_mode, sim.simx_opmode_buffer)
                # We update the list and specify that we have run the function before
                self.first_time_list[4] = False
            # We have run this function before
            else:
                # We obtain the result from coppeliasim in buffer mode
                sensor_error_list[i], sensor_resolution_list[i], sensor_image = sim.simxGetVisionSensorImage(
                    self.client_id, image_sensor_handle_list[i], color_mode, sim.simx_opmode_buffer)
            # We identify the color mode selected by the user
            if color_mode:
                # We reshape the image to the required resolution for a grayscale image
                sensor_image_list[i] = np.reshape(sensor_image, (sensor_resolution_list[i][1],
                                                                 sensor_resolution_list[i][0]))
            else:
                # We reshape the image to the required resolution for a rgb image
                sensor_image_list[i] = np.reshape(sensor_image, (sensor_resolution_list[i][1],
                                                                 sensor_resolution_list[i][0], 3))
            # The image is returned as int8, which we will convert into unsigned int8
            sensor_image_list[i] = sensor_image_list[i].astype(np.uint8)
            # We flip the image vertically as images are displayed with origin on the top normally
            sensor_image_list[i] = np.flipud(sensor_image_list[i])
        # We return the error returned for each joint of the handle list and the joint property list
        return sensor_error_list, sensor_resolution_list, sensor_image_list

    # This function helps retrieve buffers from depth sensors
    def read_depth_buffer_from_sensors(self, depth_sensor_handle_list):
        # We obtain the length of the depth_buffer sensor handle list
        sensor_list_length = len(depth_sensor_handle_list)
        # We create an empty list to hold the joint error code that will be returned for each depth_buffer sensor
        sensor_error_list = [0]*sensor_list_length
        # We create an empty list to hold the resolutions of each sensor
        sensor_resolution_list = [[0, 0]]*sensor_list_length
        # We create an empty list to hold the depth_buffers returned by each sensor
        sensor_buffer_list = [0]*sensor_list_length
        # We iterate through the proximity sensor handle list
        for i in range(sensor_list_length):
            # We check if it is the first time running the function
            if self.first_time_list[5]:
                # We obtain the result from coppeliasim in streaming mode
                _, _, _ = sim.simxGetVisionSensorDepthBuffer(self.client_id, depth_sensor_handle_list[i],
                                                             sim.simx_opmode_streaming)
                # We sleep for some time so that coppeliasim may setup the stream
                sleep(0.5)
                # We obtain the result from coppeliasim in buffer mode
                sensor_error_list[i], sensor_resolution_list[i], sensor_buffer = \
                    sim.simxGetVisionSensorDepthBuffer(self.client_id, depth_sensor_handle_list[i],
                                                       sim.simx_opmode_buffer)
                # We update the list and specify that we have run the function before
                self.first_time_list[5] = False
            # We have run this function before
            else:
                # We obtain the result from coppeliasim in buffer mode
                sensor_error_list[i], sensor_resolution_list[i], sensor_buffer = \
                    sim.simxGetVisionSensorDepthBuffer(self.client_id, depth_sensor_handle_list[i],
                                                       sim.simx_opmode_buffer)
            # We reshape the image to the required resolution for the depth buffer
            sensor_buffer_list[i] = np.reshape(sensor_buffer, (sensor_resolution_list[i][1],
                                                               sensor_resolution_list[i][0]))
            # We flip the image vertically as images are displayed with origin on the top normally
            sensor_buffer_list[i] = np.flipud(sensor_buffer_list[i])
        # We return the error returned for each joint of the handle list and the joint property list
        return sensor_error_list, sensor_resolution_list, sensor_buffer_list

# --------------- End of class definition ---------------
