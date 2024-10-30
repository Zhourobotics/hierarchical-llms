#! /usr/bin/env python

import os
import rospy
import rospkg
import numpy as np
import functools
from nav_msgs.msg import Odometry
import yaml
import openai
from std_msgs.msg import Float32MultiArray, String
from model.config_loader import ConfigLoader
import yaml, time
import re

class LLMAdaptiveServer:

    def __init__(self, config_loader, key_path):
        """
        exp_name: .yaml config file to read
        """
        with open(key_path + "/secrets.yml", 'r') as secret_file:
            secret = yaml.safe_load(secret_file)
        api_keys = secret.get('api_keys', {})
        key = api_keys[0]

        ############# General Settings #############
        self.frame_id  = config_loader.frame_id

        self.target_ids = config_loader.targetID
        self.robot_ids = config_loader.robotID
        self.dim = config_loader.dim
        self.nRobot = len(self.robot_ids)
        self.nTarget = len(self.target_ids)
        self.nTypeI = config_loader.nTypeI
        self.nTypeII = config_loader.nTypeII

        ##################### danger zone locations
        self.typeI_mu    = config_loader.typeI_mu
        self.typeI_cov   = config_loader.typeI_cov
        self.typeII_mu   = config_loader.typeII_mu
        self.typeII_cov  = config_loader.typeII_cov


        #if the folder does not exist, create it
        if not os.path.exists(key_path + "/results/data"):
            os.makedirs(key_path + "/results/data")

        self.result_path = key_path + "/results/data/" +  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".yaml"

        self.weights = config_loader.weights
        self.weights_meaning = ["control cost computed by the norm of control inputs", 
                                "tracking error computed by the trace of the estimation covariance matrix of the targets",
                                "slack variables of safety constraints to avoid sensing danger zones",
                                "slack variables of safety constraints to avoid communication danger zones"]
        self.task_ability = config_loader.task_ability
        self.assignment_matrix = config_loader.assignment_matrix


        self.cur_weights = self.weights.copy()
        self.cur_assignment_matrix = self.assignment_matrix.copy()

         
        # llm model: gpt-3.5-turbo, gpt-4o-2024-08-06, gpt-4o-mini
        self.llm_model = "gpt-3.5-turbo"
        self.max_token = int(50 * (self.task_ability + 2))
        self.temperature = 0

        if self.llm_model  == "gpt-4o-2024-08-06":
            self.max_token += 200



        self.llm_inner_dur = config_loader.llm_inner_dur
        self.llm_outer_dur = config_loader.llm_outer_dur
        self.results = {}

        self.client_task = openai.OpenAI(api_key=key)
        self.client_tuning = openai.OpenAI(api_key=key)


        ######### data #########
        self.his_sub_robot_odom_all = []
        self.his_sub_target_odom_all = []

        for i in range(len(self.robot_ids)):
            self.his_sub_robot_odom_all.append([])
        for i in range(len(self.target_ids)):
            self.his_sub_target_odom_all.append([])

        if config_loader.exp == "ros simulation" or config_loader.exp == "ros real":
            from tracker.msg import Results
            self.result_sub = rospy.Subscriber("server/results", Results, self.result_callback)

            self.assignment_matrix_pub = rospy.Publisher("server/assignment_matrix", Float32MultiArray, queue_size=10)
            self.weights_pub = rospy.Publisher("server/weights", Float32MultiArray, queue_size=10)
            
            self.timer_assign_callback =  rospy.Timer(rospy.Duration(self.llm_outer_dur), self.timer_assign_callback)
            self.timer_weight_callback =  rospy.Timer(rospy.Duration(self.llm_inner_dur), self.timer_weight_callback)


            self.human_inputs_sub = rospy.Subscriber("server/human_inputs", String, self.human_inputs_callback)
            self.llm_output_pub   = rospy.Publisher("server/llm_outputs", String, queue_size=10)
            self.llm_prompt_pub   = rospy.Publisher("server/llm_prompts", String, queue_size=10)
            self.evaluation_pub   = rospy.Publisher("server/evaluation", String, queue_size=10)


            # rospy.init_node('llm_server', anonymous=False)
            # rate = rospy.Rate(10)
            rospy.init_node('llm_server', anonymous=True)

            # while not rospy.is_shutdown():
            #     rate.sleep()
        ######### solver data #########
        self.his_results = []
        self.his_drone_pos        = []
        self.his_target_pos       = []


        self.his_task_prompts     = []
        self.his_task_assignments = []
        self.his_task_outputs     = []
        self.his_correct_flags    = []


        self.his_weights_prompts  = []
        self.his_weights          = []
        self.his_weights_outputs   = []
        self.his_weights_correct_flags = []

        

        ################ evaluation data ################

        #### 1. for task assignment
        self.task_total_call = 0
        self.task_correct_call = 0
        self.task_correct_rate = 0

        self.task_total_token = 0
        self.task_avg_token = 0

        self.task_total_prompt_token = 0
        self.task_avg_prompt_token = 0

        self.task_total_time = 0
        self.task_avg_time = 0


        self.weights_total_call = 0
        self.weights_correct_call = 0
        self.weights_correct_rate = 0

        self.weights_total_prompt_token = 0
        self.weights_avg_prompt_token = 0

        self.weights_total_token = 0
        self.weights_avg_token = 0

        self.weights_total_time = 0
        self.weights_avg_time = 0


        ########### result data ###########
        self.total_trace = 0
        self.total_avg_trace = 0

        self.total_attacked = 0



    def update_results(self, results):
        self.results = results.copy()
        self.his_results.append(results.copy())

        self.total_trace += results["trace"].copy()
        self.total_avg_trace = self.total_trace / len(self.his_results)
        self.total_attacked += np.sum(results["attacked_typeI_flags"]) + np.sum(results["attacked_typeII_flags"])


        return


    def evaluation(self):

        evaluation = "Task assignment evaluation: \n"
        evaluation += "Total call: " + str(self.task_total_call) + ", "
        evaluation += "Correct call: " + str(self.task_correct_call) + ", "
        evaluation += "Correct rate: " + str(self.task_correct_rate) + ", "
        evaluation += "Average token number: " + str(self.task_avg_token) + ","
        evaluation += "Average prompt token number: " + str(self.task_avg_prompt_token) + ". \n"


        evaluation += "Weight tuning evaluation: \n"
        evaluation += "Total call: " + str(self.weights_total_call) + ", "
        evaluation += "Correct call: " + str(self.weights_correct_call) + ", "
        evaluation += "Correct rate: " + str(self.weights_correct_rate) + ", "
        evaluation += "Average token number: " + str(self.weights_avg_token) + ","
        evaluation += "Average prompt token number: " + str(self.weights_avg_prompt_token) + ". \n"

        evaluation += "Total task time: " + str(self.task_total_time) + ", "
        evaluation += "Average task time: " + str(self.task_avg_time) + ". \n"
        evaluation += "Total weight time: " + str(self.weights_total_time) + ", "
        evaluation += "Average weight time: " + str(self.weights_avg_time) + ". \n"


        ### result data
        evaluation += "Total trace: " + str(self.total_trace) + ", "
        evaluation += "Average trace: " + str(self.total_avg_trace) + ". \n"
        evaluation += "Total attacked: " + str(self.total_attacked) + ". \n"

        self.evaluation_pub.publish(String(evaluation))
        return


    def save_data(self, path = ""):

        if path == "":
            path = self.result_path

        #print("Save data to: ", path)

        ## in json format
        his_data = []
        data = {"robot_pos": [], "target_pos": [],
                "weights_prompts": "", "weights": [],
                "weights_outputs": "" , "weights_correct_flags": []}
                 
        task_data = {"task_prompts": "", "task_assignments": [], "task_outputs": "", "task_correct_flags": []}

        if len(self.his_weights_prompts) <= 1:
            print("No data to save. ")
            return 




        for i in range(len(self.his_weights_prompts)):

            ## only keep two decimal points of the pos
            data["robot_pos"] = np.round(self.his_drone_pos[i].T, 2).tolist()
            data["target_pos"] = np.round(self.his_target_pos[i].T, 2).tolist()
            data["weights_prompts"] = self.his_weights_prompts[i]
            #print("his_weights is: ", self.his_weights[i])
            data["weights"] = self.his_weights[i]
            data["weights_outputs"] = self.his_weights_outputs[i]
            data["weights_correct_flags"] = self.his_weights_correct_flags[i]
            his_data.append(data.copy())

        if len(self.his_task_prompts) >= 1:

            for i in range(len(self.his_task_prompts)):
                task_data["task_prompts"] = self.his_task_prompts[i]
                task_data["task_assignments"] = self.his_task_assignments[i].tolist()
                task_data["task_outputs"] = self.his_task_outputs[i]
                task_data["task_correct_flags"] = self.his_correct_flags[i]
                his_data.append(task_data.copy())

            
        # evaluation data
        evaluation = {"task_total_call": self.task_total_call,
                        "task_correct_call": self.task_correct_call,
                        "task_correct_rate": self.task_correct_rate,
                        "task_total_token": self.task_total_token,
                        "task_avg_token": self.task_avg_token,
                        "task_total_time": self.task_total_time,
                        "task_avg_time": self.task_avg_time,
                        "task_total_prompt_token": self.task_total_prompt_token,
                        "task_avg_prompt_token": self.task_avg_prompt_token,
                        "weights_total_call": self.weights_total_call,
                        "weights_correct_call": self.weights_correct_call,
                        "weights_correct_rate": self.weights_correct_rate,
                        "weights_total_token": self.weights_total_token,
                        "weights_avg_token": self.weights_avg_token,
                        "weights_total_time": self.weights_total_time,
                        "weights_avg_time": self.weights_avg_time,
                        "weights_total_prompt_token": self.weights_total_prompt_token,
                        "weights_avg_prompt_token": self.weights_avg_prompt_token,
                        "total_trace": self.total_trace,
                        "total_avg_trace": self.total_avg_trace,
                        "total_attacked": self.total_attacked}
        

        his_data.append(evaluation)

        #print("his_data is: ", his_data)
        with open(path, 'w') as file:
            for i in range(len(his_data)):
                ##add a header
                file.write("Data " + str(i) + ":\n")
                for key in his_data[i]:
                    if type(his_data[i][key]) == str:
                        file.write("  " + key + ": \"" + his_data[i][key] + " \"\n")
                    else:
                        file.write("  " + key + ": " + str(his_data[i][key]) + "\n")

        return
    
    
    def publish_task_results(self, assignment_matrix):

        msg = Float32MultiArray()
        msg.data = assignment_matrix.flatten()

        print("timer assignment matrix is: ", assignment_matrix)
        self.assignment_matrix_pub.publish(msg)

        if len(self.his_task_outputs) > 0:
            self.llm_output_pub.publish(String(self.his_task_outputs[-1]))
            self.llm_prompt_pub.publish(String(self.his_task_prompts[-1]))
        return

    def publish_weight_results(self, weights):

        msg = Float32MultiArray()
        msg.data = weights

        print("timer weights is: ", weights)
        self.weights_pub.publish(msg)

        if len(self.his_weights_outputs) > 0:
            self.llm_output_pub.publish(String(self.his_weights_outputs[-1]))
            self.llm_prompt_pub.publish(String(self.his_weights_prompts[-1]))
        return
    



    def human_inputs_callback(self, data):

        #print("human inputs is: ", data.data)
        inputs = data.data

        prompt = "The human supervisor has inputs: " + inputs
        print("human" )

        assignment_matrix = self.task_adapter(additional_prompt = prompt)
        weights = self.weight_adapter(additional_prompt = prompt)

        self.publish_task_results(assignment_matrix)
        self.publish_weight_results(weights)
        self.evaluation()
        return


    def timer_assign_callback(self, event):
        assignment_matrix = self.task_adapter()
        self.publish_task_results(assignment_matrix)
        self.evaluation()
        return
    
    def timer_weight_callback(self, event):
        #update the tracker manager
        weights = self.weight_adapter()
        self.publish_weight_results(weights)
        self.evaluation()
        return
    







    def result_callback(self, data):

        # std_msgs/Header header
        # int32 id
        # geometry_msgs/Point[] robot_pos
        # geometry_msgs/Point[] robot_vel
        # geometry_msgs/Point[] target_pos
        # geometry_msgs/Point[] robot_cmd
        # int32 exitflag
        # int32[] known_typeI_flags
        # int32[] known_typeII_flags
        # int32[] attacked_typeI_flags
        # int32[] attacked_typeII_flags
        # int32[] trace
        # results = {"robot_pos": [], "robot_vel": [], 
        #            "target_pos": [], "robot_cmd": [], 
        #            "exitflag": [], 
        #            "known_typeI_flags": [], "known_typeII_flags": [], 
        #            "attacked_typeI_flags": [], "attacked_typeII_flags": [],
        #            "trace": [], "weights": [], "assignment_matrix": []}
        results = {}
        robot_pos           = np.zeros((self.nRobot, self.dim))
        robot_cmd           = np.zeros((self.nRobot, self.dim))
        robot_vel           = np.zeros((self.nRobot, self.dim))

        for i in range(self.nRobot):
            robot_pos[i] = np.array([data.robot_pos[i].x, data.robot_pos[i].y])
            robot_cmd[i] = np.array([data.robot_cmd[i].x, data.robot_cmd[i].y])
            if len(data.robot_vel) > 0:
                robot_vel[i] = np.array([data.robot_vel[i].x, data.robot_vel[i].y])
        results["robot_pos"] = robot_pos
        results["robot_cmd"] = robot_cmd
        results["robot_vel"] = robot_vel

        target_pos        = np.zeros((self.nTarget, self.dim))
        for i in range(self.nTarget):
            target_pos[i] = np.array([data.target_pos[i].x, data.target_pos[i].y])
        results["target_pos"] = target_pos

        results["exitflag"] = data.exitflag
        

        # flags for if knows the danger zones
        known_typeI_flags      = np.zeros((self.nRobot, self.nTypeI))
        known_typeII_flags     = np.zeros((self.nRobot, self.nTypeII))

        # flags for if attacked integer type
        attacked_typeI_flags  = np.zeros((self.nRobot))    # don't care attacked by which type I zone
        attacked_typeII_flags = np.zeros((self.nRobot))     # don't care attacked by which type II zone

        for i in range(self.nRobot):
            attacked_typeI_flags[i] = data.attacked_typeI_flags[i]
            attacked_typeII_flags[i] = data.attacked_typeII_flags[i]
            for j in range(self.nTypeI):
                known_typeI_flags[i][j] = data.known_typeI_flags[i*self.nTypeI + j]
            for j in range(self.nTypeII):
                known_typeII_flags[i][j] = data.known_typeII_flags[i*self.nTypeII + j]
        
        results["known_typeI_flags"] = known_typeI_flags
        results["known_typeII_flags"] = known_typeII_flags
        results["attacked_typeI_flags"] = attacked_typeI_flags
        results["attacked_typeII_flags"] = attacked_typeII_flags

        results["trace"] = data.trace

        results["weights"] = data.weights
        assignment_matrix = np.zeros((self.nRobot, self.nTarget))
        for i in range(self.nRobot):
            for j in range(self.nTarget):
                assignment_matrix[i][j] = data.assignment_matrix[i*self.nTarget + j]
        results["assignment_matrix"] = assignment_matrix

        #print("results is: ", results)
        self.results = results.copy()
        self.his_results.append(results.copy())
        return


    def drone_odom_callback(self, data, index):
        """
        data: Crazyswarm GenericLogData
        id: int, id of the drone
        """
        # extract the position
        x, y, z = data.values[0], data.values[1], data.values[2]
        # update to the tracker manager
        self.his_sub_robot_pos_all[index].append(np.array([x, y]))

        return

    def target_odom_callback(self, data, index):

        x, y, z = data.values[0], data.values[1], data.values[2]
        
        self.his_sub_target_pos_all[index].append(np.array([x, y]))
        return
        
    # 1 ~ 2 seconds
    def generate_task_assignment_prompt(self):

        #generate the prompt"
        prompt = self.interpolate_history_results()
        prompt += "Each drone has ability to track at most " + str(self.task_ability) + " targets, "
        prompt += "and each target should be tracked by at least one drone if possible. "

        prompt += "The last assignment matrix is: "
        
        if len(self.his_task_assignments) == 0:
            last_assignment = self.assignment_matrix
        else:
            last_assignment = self.his_task_assignments[-1]


        for i in range(len(self.robot_ids)):
            prompt += "Drone " + str(self.robot_ids[i]) + " is assigned to track "
            for j in range(len(self.target_ids)):
                if last_assignment[i][j] == 1:
                    prompt += "Target " + str(self.target_ids[j]) + ", "

            # delete the last comma
            prompt = prompt[:-2]
            prompt += ". "



        prompt += "Please provide a new tracking assignment for each drone as a list of target id each sentence. "
        prompt += "And add feedback to the human supervisor. "
        return prompt
    

    def interpolate_history_results(self, n = 5):

        #print("n is: ", len(self.his_results))
        #print("num_temp is: ", num_temp)
        #print("num temp type is: ", type(num_temp))

        num = min(n, int(len(self.his_results)))
        prompt = "The recent " + str(num) + " results of status and observations are: "
        for i in range(num):
            prompt += "The " + str(i+1) + "th information is as follows. "
            results = self.his_results[-i]
            prompt += self.interpolate_results(results)
            prompt += " \n"


        return prompt

    def interpolate_positions(self, results):
        
        ################# robot current information
        prompt = "Drones are currently at the following positions: "
        for i in range(len(results["robot_pos"])):
            id = self.robot_ids[i]
            prompt += "Drone " + str(id) + " is at " + str(results["robot_pos"][i]) + ". "
        
        prompt += "Targets are currently at the following positions: "
        for i in range(len(results["target_pos"])):
            id = self.target_ids[i]
            prompt += "Target " + str(id) + " is at " + str(results["target_pos"][i]) + ". "
        
        return prompt
    

    def interpolate_results(self, results):
        """
        interpolate the results
        """

        ################# robot current information
        prompt = "Drones are currently at the following positions: "
        for i in range(len(results["robot_pos"])):
            drone_id = self.robot_ids[i]
            rounded_pos = np.round(results["robot_pos"][i], 5)
            prompt += "Drone " + str(drone_id) + " is at " + str(rounded_pos) + ". "
        
        prompt += "Targets are currently at the following positions: "
        for i in range(len(results["target_pos"])):
            target_id = self.target_ids[i]
            rounded_pos = np.round(results["target_pos"][i], 5)
            prompt += "Target " + str(target_id) + " is at " + str(rounded_pos) + ". "
        

        ################# known dangerous zones 
        if self.nTypeI > 0:
            has_known_sensing = False
            sensing_known_prompt = "The known sensing zones are: "
            for jZone in range(self.nTypeI):
                zone_prompt = "The sensing zone " + str(jZone) + " is located at " + str(self.typeI_mu[jZone]) + " with covariance " + str(self.typeI_cov[jZone]) + ". "
                zone_j_known = False
                for i in range(len(results["known_typeI_flags"])):
                    if results["known_typeI_flags"][i][jZone] == 1:
                        drone_id    = self.robot_ids[i]
                        zone_prompt += "Drone " + str(drone_id) + " knows the sensor zone " + str(jZone) + ". "
                        has_known_sensing = True
                        zone_j_known = True
                if zone_j_known == True:
                    sensing_known_prompt += zone_prompt

            if has_known_sensing == False:
                sensing_known_prompt = "No sensing zone has been detected. "
            prompt += sensing_known_prompt
        else:
            prompt += "No sensing zone is in the environment. "

        if self.nTypeII > 0:     
            has_known_communication = False
            comm_known_prompt = "The known communication zones are: "
            for jZone in range(self.nTypeII):
                zone_prompt = "The communication zone " + str(jZone) + " is located at " + str(self.typeII_mu[jZone]) + " with covariance " + str(self.typeII_cov[jZone]) + ". "
                zone_j_known = False
                for i in range(len(results["known_typeII_flags"])):
                    if results["known_typeII_flags"][i][jZone] == 1:
                        drone_id    = self.robot_ids[i]
                        zone_prompt += "Drone " + str(drone_id) + " knows the communication zone " + str(jZone) + ". "
                        has_known_communication = True
                if zone_j_known == True:
                    comm_known_prompt += zone_prompt

            if has_known_communication == False:
                comm_known_prompt = "No communication zone has been detected. "
            prompt += comm_known_prompt
        else:
            prompt += "No communication zone is in the environment. "


        ################# the attack information
        attack_prompt = "The attack information is: "
        has_attack = False
        for i in range(len(results["attacked_typeI_flags"])):
            if results["attacked_typeI_flags"][i] == 1:
               drone_id = self.robot_ids[i]
               attack_prompt += "The drone " + str(drone_id) + " has sensor failure. "
               has_attack = True
        for i in range(len(results["attacked_typeII_flags"])):
            if results["attacked_typeII_flags"][i] == 1:
               drone_id = self.robot_ids[i]
               attack_prompt += "The drone " + str(drone_id) + " has communication failure. "
               has_attack = True
        
        if has_attack == False:
            attack_prompt = "No attack has been detected. "

        prompt += attack_prompt

        ################## trace
        prompt += "The trace of the tracking estimation error covariances matrix is "
        prompt += str(np.round(results["trace"], 5)) + " (smaller is better). "

        #print("Prompt is: ", prompt)
        

        return prompt


    def extract_task_assignment_results(self, completion):

        assignment_matrix = np.zeros((len(self.robot_ids), len(self.target_ids)))
        results = completion.choices[0].message.content

        # results = "Drone 0: Assign to Target 0 and Target 1  \nDrone 1: Assign to Target 2 and Target 3  \nDrone 2: Assign to Target 0 and Target 2"
        # # formatis Drone 0: Assign to Target 0 and Target 1  \n
        # #  Drone 1: Assign to Target 2 and Target 3  \nDrone 2: Assign to Target 0 and Target 2
        # #exact_position = json.loads(re.findall(pattern, self.latest.split("\nTarget: ")[1])[-1])
        #print("results is: ", results)
        import re 
        import json

        #res = json.loads(results)
        #convert the string to dictionary
        #results = res
        #results = results["Assignment matrix"]
        print("results is: ", results)
        results = results.split("\n")

        ##find the start of the sentence by finding the drone id
        drone_num = 0
        for i in range(len(results)):
            if "Drone" in results[i]:
                #print("results[i] is: ", results[i])
                # result is start with "Drone"
                

                # check if only one Drone in this sentence
                num = re.findall(r'Drone \d+', results[i])
                if len(num) != 1:
                    print("Warning: multiple drones in one sentence. ")
                    continue


                result = re.findall(r'Drone \d+', results[i])
                drone_id = int(re.findall(r'\d+', result[0])[0])
                drone_order = -1
                for j in range(len(self.robot_ids)):
                    if drone_id == self.robot_ids[j]:
                        drone_order = j
                        break
                # find the target ids
                target_result = re.findall(r'Target \d+', results[i])
                if len(target_result) == 0:
                    #target_result = re.findall(r'\d+', results[i])
                    continue
                #print("result is: ", target_result)
                for j in range(len(target_result)):
                    #print("target_result[j] is: ", target_result[j])
                    target_id = int(re.findall(r'\d+', target_result[j])[0])
                    #print("target_id is: ", target_id)
                    #print("drone id is ", drone_id)
                    for k in range(len(self.target_ids)):
                        if target_id == self.target_ids[k]:
                            assignment_matrix[drone_order][k] = 1
                    # else:
                    #     print("Warning: target id is not in the target list. ")
                    #     print("target id is: ", target_id)
                    #     print("target list is: ", self.target_ids)
                drone_num += 1
            if drone_num == len(self.robot_ids):
                break
                

        return assignment_matrix
            

    def task_adapter(self, additional_prompt = ""):
        """
        update the drone and target positions
        """
        if self.results == {}:
            print("no results received, just return the previous assignment matrix")
            return self.assignment_matrix
        time1 = time.time()
        prompt = self.generate_task_assignment_prompt()
        # sample = openai.completions.create(engine='text-davinci-002',
        #                                    prompt=prompt,
        #                                    max_tokens=100,
        #                                    temperature=0)
        prompt += additional_prompt


        completion = \
        self.client_task.chat.completions.create(model=self.llm_model, 
                                                 messages = [{"role": "system", 
                                                            "content": "You are an optimizer with the goal of assigning tracking targets for drones. "},
                                                            {
                                                            "role": "user",
                                                            "content": prompt},
                                                            ],  
                                                 max_tokens=self.max_token,
                                                 temperature=self.temperature)
        

        
        #print("GPT3 output is: ", completion)
        assignment_matrix = self.extract_task_assignment_results(completion)
        time2 = time.time()
        # check the token number

        # Access the token usage
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens


        print("[task_adapter] total tokens is: ", total_tokens)
        print("[task_adapter] prompt tokens is: ", prompt_tokens)
        print("[task_adapter] completion tokens is: ", completion_tokens)



        ################# task certificates #################
        #print("assignment_matrix is: ", assignment_matrix)
        is_feasible = True
        for i in range(len(self.target_ids)):
            #print("assignment_matrix[:, i] is: ", assignment_matrix[:, i])
            if np.sum(assignment_matrix[:, i]) == 0:
                id = self.target_ids[i]
                print("Warning: target ", id, " is not assigned to any drone. The llm is not feasible. ")
                assignment_matrix[:, i] = self.assignment_matrix[:, i]
                is_feasible = False
        
        for i in range(len(self.robot_ids)):
            if np.sum(assignment_matrix[i]) > self.task_ability:
                id = self.robot_ids[i]
                print("Warning: drone ", id, " is assigned to more than ", self.task_ability, " targets. The llm is not feasible. ")
                assignment_matrix[i, :] = self.assignment_matrix[i, :]
                is_feasible = False


        ################# save data #################
        self.his_task_prompts.append(prompt)
        self.his_task_assignments.append(assignment_matrix.copy())
        self.his_task_outputs.append(completion.choices[0].message.content)
        self.his_correct_flags.append(is_feasible)

        ################# evaluation #################
        self.task_total_call += 1
        self.task_total_token += completion_tokens
        self.task_correct_call += is_feasible

        self.task_total_prompt_token += prompt_tokens
        self.task_avg_prompt_token = self.task_total_prompt_token / self.task_total_call


        self.task_avg_token = self.task_total_token / self.task_total_call
        self.task_correct_rate = self.task_correct_call / self.task_total_call

        self.task_total_time += time2 - time1
        self.task_avg_time = self.task_total_time / self.task_total_call

        return (assignment_matrix).copy()

    


    def generate_weight_prompt(self, results):

        prompt = self.interpolate_results(results)
        prompt += "The current weights for objective functions are: "
        for i in range(len(results["weights"])):
            prompt += "The weight " + str(i) + " is " + self.weights_meaning[i] + ", "
            prompt += "and the value is " + str(np.round(results["weights"][i], 5)) + ". "


        prompt += "You should provide a new weight vector as a list format inside '[]' with a length of " + str(len(self.weights)) + "."
        prompt += "And add description feedback to human supervisor in a new paragraph. "



        #print("Prompt is: ", prompt)
        return prompt
    


    def weight_adapter(self, additional_prompt = ""):

        if self.results == {}:
            print("no results received, just return the previous weights")
            return self.weights

        time1 = time.time()
        prompt = self.generate_weight_prompt(self.results)
        prompt += additional_prompt
        
        completion = \
        self.client_tuning.chat.completions.create(model=self.llm_model,
                                                   messages = [{"role": "system", 
                                                        "content": "You are a multiple objective optimizer with the goal of specifying the weights of each objective function. "},
                                                        {
                                                        "role": "user",
                                                        "content": prompt},
                                                        ],  
                                                    max_tokens=self.max_token,
                                                    temperature=self.temperature)
        #print("[weight_adapter] GPT3 output is: ", completion)

        # response = self.client.embeddings.create(
        #     model="text-embedding-3-large",
        #     input=completion.choices[0].message.content
        # 
        llm_output = completion.choices[0].message.content
        time2 = time.time()
        # check the token number

        # Access the token usage
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

        print("[weight_adapter] total tokens is: ", total_tokens)
        print("[weight_adapter] prompt tokens is: ", prompt_tokens)
        print("[weight_adapter] completion tokens is: ", completion_tokens)


        token_num = completion_tokens


        #results =  Here is a new weight vector suggestion in the form of a Python list:
        # [0.02, 7.0, 150.0, 150.0]
        # vector is the value inside [ ]
        import ast  # For safe literal evaluation

        is_valid = True
        weights = None
        try:
            vector_str = re.findall(r'\[.*\]', llm_output)[0]
            # Convert string representation of a list to an actual list
            vector = ast.literal_eval(vector_str)  # safer than eval
            
            # Now convert the list to a numpy array
            weights = np.array(vector)
            print("[weight_adapter] The adjust weights is: ", weights)

        except:
            print("[weight_adapter] Error: the output is not in the right format")
            print("[weight_adapter] The output is: ", llm_output)

            try:
                data_str = re.findall(r'\[.*\]', llm_output)[0]
                data = ast.literal_eval(data_str)

                print("data is: ", data)
                if len(data) >= len(self.weights):
                    weights = np.array(data)[:len(self.weights)]
                    print("weights is: ", weights)
                else:
                    is_valid = False
                    weights = np.array(self.weights.copy())
            except:
                is_valid = False
                weights = np.array(self.weights.copy())

        # check if the weights are valid
        for i in range(len(weights)):
            if weights[i] < 0:
                print("Warning: the weight ", i, " is negative. The weight is set to the previous value. ")
                is_valid = False
                weights[i] = self.weights[i]
            if weights[i] > 1e4:
                print("Warning: the weight ", i, " is too large. The weight is set to the previous value. ")
                is_valid = False
                weights[i] = self.weights[i]

        ################# save data #################
        #print("result robot_pos is: ", results["robot_pos"])
        self.his_drone_pos.append(self.results["robot_pos"])
        self.his_target_pos.append(self.results["target_pos"])
        self.his_weights_prompts.append(prompt)
        self.his_weights.append(weights.copy())
        self.his_weights_outputs.append(completion.choices[0].message.content)
        self.his_weights_correct_flags.append(is_valid)

        ################# evaluation #################
        self.weights_total_call += 1
        self.weights_total_token += token_num
        self.weights_correct_call += is_valid

        self.weights_total_prompt_token += prompt_tokens
        self.weights_avg_prompt_token = self.weights_total_prompt_token / self.weights_total_call


        self.weights_avg_token = self.weights_total_token / self.weights_total_call
        self.weights_correct_rate = self.weights_correct_call / self.weights_total_call

        self.weights_total_time += time2 - time1
        self.weights_avg_time = self.weights_total_time / self.weights_total_call


        return weights.copy()

    

    def ros_adapter(self):
        """
        setup ROS parameters
        """

        rospy.init_node('tracker_server', anonymous=True)
        rate = rospy.Rate(1/self.config_loader.dt)

        #subscribe the robot information, the dangerous zone information
        for i in range(len(self.robot_ids)):
            self.drone_odom_subs.append(rospy.Subscriber("/drone" + str(self.robot_ids[i]) + "/odom",
                                        Odometry, functools.partial(self.drone_odom_callback, index=i)))

        for i in range(len(self.target_ids)):
            self.target_odom_subs.append(rospy.Subscriber("/target" + str(self.target_ids[i]) + "/odom",
                                        Odometry, functools.partial(self.target_odom_callback, index=i)))

        
        return







if __name__ == '__main__':
    
    rospy.init_node('llm_server', anonymous=True)
    exp_name = rospy.get_param('~exp_name')

    print("exp_name is: ", exp_name)



    try :
        path = rospkg.RosPack().get_path('tracker')
    except:
        #use relative path
        path = os.path.dirname(os.path.abspath(__file__)) + "/.."
    
    print("path is: ", path)
    #path = "/home/grasp/ma_ws/src/target_tracking/src/tracker"
    config_path = path + "/config/" + exp_name + ".yaml"
    if not os.path.exists(config_path):
        print("Config file does not exist")
    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config_loader = ConfigLoader(config)
    tracker_server = LLMAdaptiveServer(config_loader, path)
    rospy.spin()
