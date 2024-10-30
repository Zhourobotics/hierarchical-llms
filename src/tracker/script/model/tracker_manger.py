import numpy as np
import yaml
from model.problem import Problem
from model.danger_zones import DangerZones
from model.forcepro_centralized import ForcesProSolverCent
from model.forcepro_single import ForcesProSolverSingle
from scipy.stats import norm
import sys
import math


class TrackerManager:

    def __init__(self, config_loader):
        """
        config_path: .yaml config file to read
        """

        ############# General Settings #############
        self.testID = config_loader.testID
        self.steps = config_loader.steps
        self.exp = config_loader.exp
        self.frame_id = config_loader.frame_id
        print("frame_id = ", self.frame_id)
        ############# Danger Zone and Problem #############
        self.trac_prob    = Problem(config_loader) # just get the config
        self.danger_zones = DangerZones(config_loader)

        
        ############# Problem #############
        self.nRobot = self.trac_prob.nRobot
        self.nTarget = self.trac_prob.nTarget
        self.dt = self.trac_prob.dt
        self.dim = self.trac_prob.dim
        self.N = self.trac_prob.N # horizon(stages)]
        
        
        ############# Centrailized Solver and Single Solver #############
        self.use_cent_solver = config_loader.use_cent_solver


        self.forcespro = ForcesProSolverCent(self.trac_prob, self.danger_zones)
        self.cent_solver = self.forcespro.solver

        ### if write solver in one file 
        ### each robot has one solver
        self.single_solver = []
        self.single_forcespro = []
        for i in range(self.nRobot):
            single_forcespro = ForcesProSolverSingle(self.trac_prob, self.danger_zones, i)
            self.single_solver.append(single_forcespro.solver)
            self.single_forcespro.append(single_forcespro)
        

        ############# Initial State #############
        self.state = np.zeros((self.nRobot, self.dim))
        self.cmd = np.zeros((self.nRobot, self.dim))


        ############# Flags #############
        self.cur_step = 0
        self.last_exitflag = 1

        # flags for if knows the danger zones
        self.known_typeI_flags      = np.zeros((self.nRobot, self.danger_zones.nTypeI))
        self.known_typeII_flags     = np.zeros((self.nRobot, self.danger_zones.nTypeII))

        # flags for if attacked integer type
        self.attacked_typeI_flags  = np.zeros((self.nRobot))    # don't care attacked by which type I zone
        self.attacked_typeII_flags = np.zeros((self.nRobot))     # don't care attacked by which type II zone

        # flags for if inside the danger zones
        self.inside_typeI_flags = np.zeros((self.nRobot, self.danger_zones.nTypeI))
        self.inside_typeII_flags = np.zeros((self.nRobot, self.danger_zones.nTypeII))



        ############# Log and data #############
        self.typeI_attacked_pos = []
        self.typeII_attacked_pos = []
        self.trace_list = []
        self.trace_list_single = []
        for i in range(self.nRobot):
            self.trace_list_single.append([])



    def get_robot_with_attack(self):
        
        return (self.attacked_typeI_flags == 1) | (self.attacked_typeII_flags == 1)

    def get_known_danger_zones(self):
        """
        get the known danger zone positions
        """
        known_typeI_zones = {"typeI_mu": [], "typeI_cov": []}
        known_typeII_zones = {"typeII_mu": [], "typeII_cov": []}

        for i in range(self.danger_zones.nTypeI):
            if self.known_typeI_flags[:, i].all():
                known_typeI_zones["typeI_mu"].append(self.danger_zones.typeI_mu[i])
                known_typeI_zones["typeI_cov"].append(self.danger_zones.typeI_cov[i])
        for i in range(self.danger_zones.nTypeII):
            if self.known_typeII_flags[:, i].all():
                known_typeII_zones["typeII_mu"].append(self.danger_zones.typeII_mu[i])
                known_typeII_zones["typeII_cov"].append(self.danger_zones.typeII_cov[i])
        return known_typeI_zones, known_typeII_zones

    
    ############# yuwei: read this function #############
    def update_attacked_robots(self):
        """
        update the attacked robots
        """
        self.inside_typeI_flags  = self.danger_zones.if_in_typeI_zones(self.trac_prob.robot_odom)
        self.inside_typeII_flags = self.danger_zones.if_in_typeII_zones(self.trac_prob.robot_odom)
        
        # step one check each robot if in the zone
        for iRobot in range(self.nRobot):
            robot_pos = self.trac_prob.robot_odom[iRobot, :self.dim]
            is_new_attack = False

            ################### type II ###################
            for iZone in range(self.danger_zones.nTypeI):
                ################# attack process #################
                # the robot can get multiple attack in one region
                if self.danger_zones.attack_model(robot_pos, iZone, "typeI", self.dt):

                    #if self.inside_typeI_flags[iRobot, iZone] == 1:
                    self.attacked_typeI_flags[iRobot] = 1
                    is_new_attack = True
                    print("\033[91m ---------- Robot {} is attacked by type I zone {} \033[0m".format(iRobot, iZone))
                    self.typeI_attacked_pos.append(robot_pos)

                    # if get attacked, then update the known zone
                    # update the known zone for all robots in the communication network
                    self.known_typeI_flags[iRobot, iZone] = 1
                    if self.use_cent_solver == True:
                        self.known_typeI_flags[self.attacked_typeII_flags[:] == 0, iZone] = 1

                    continue


            ################# recover process
            #print("self.inside_typeI_flags = ", self.inside_typeI_flags)
            if is_new_attack == False and \
                self.danger_zones.recover_model_w_inside(self.inside_typeI_flags[iRobot, :]):
                if self.attacked_typeI_flags[iRobot] == 1:
                    print("\033[93m robot {} is recovered from type I zone \033[0m".format(iRobot))
                    self.attacked_typeI_flags[iRobot] = 0

                    self.trac_prob.merge_estimation_to_central(iRobot, self.get_robot_with_attack())


            is_new_attack = False
            ################### type II ###################
            for iZone in range(self.danger_zones.nTypeII):
                ################# attack process #################
                if self.danger_zones.attack_model(robot_pos, iZone, "typeII", self.dt) or \
                self.danger_zones.jamming_comm_model(robot_pos, 
                                                        self.trac_prob.robot_odom[:, :self.dim], 
                                                        iZone):
                

                    #if self.inside_typeII_flags[iRobot, iZone] == 1:
                    self.attacked_typeII_flags[iRobot] = 1
                    is_new_attack = True
                    print("\033[94m robot {} is attacked by type II zone {} \033[0m".format(iRobot, iZone))
                    self.typeII_attacked_pos.append(robot_pos)

                    # communication attack cannot broadcast to other robots instantly
                    self.known_typeII_flags[iRobot, iZone] = 1
                    continue

            ################# recover process
            ##self.danger_zones.recover_comm_model(robot_pos, self.trac_prob.robot_odom):
            if is_new_attack == False and \
                self.danger_zones.recover_model_w_inside(self.inside_typeII_flags[iRobot, :]):
                if self.attacked_typeII_flags[iRobot] == 1:
                    print("\033[93m robot {} is recovered from type II zone. \033[0m".format(iRobot))
                    self.attacked_typeII_flags[iRobot] = 0
                    # when it get recovered, then update the known zone
                    known_zones = np.where(self.known_typeII_flags[iRobot, :] == 1)[0]

                    if self.use_cent_solver == True:
                        for k_zone in known_zones:
                            print(self.attacked_typeII_flags[:] == 0)
                            print("k_zone = ", k_zone)
                            print("known_zones = ", known_zones)
                            self.known_typeII_flags[self.attacked_typeII_flags[:] == 0, k_zone] = 1


        
        # print("attacked_typeI_flags = ", self.attacked_typeI_flags)
        # print("attacked_typeII_flags = ", self.attacked_typeII_flags)
        return 
                         



    def update_running_params(self, problem_setup, com_attack_idx = -1):
        """
        update new best est pos and voc for targets
        """
        next_best_est_tar_pos = np.copy(self.trac_prob.targets_best_est_pos)
        next_best_est_tar_cov = np.copy(self.trac_prob.targets_best_est_cov)
        stage_num = self.forcespro.Lrvars_total



        if com_attack_idx != -1:
          stage_num = self.single_forcespro[0].Lrvars_total
          next_best_est_tar_pos = np.copy(self.trac_prob.targets_best_est_pos_single[com_attack_idx])
          next_best_est_tar_cov = np.copy(self.trac_prob.targets_best_est_cov_single[com_attack_idx])


        running_params = np.zeros((stage_num * self.N, 1))
        #print("running_params_size at each stage = ", stage_num)

        
        for iStage in range(self.N):
            #print("iStage = ", iStage)
            idx = iStage * stage_num
             
            ################### part 1: robot state and command ###################
            for jTarget in range(self.nTarget):
                start_idx = idx + jTarget * self.trac_prob.Lrtargets

                #print("target {} best est pos = ".format(jTarget), next_best_est_tar_pos[jTarget, :, iStage])
                running_params[start_idx: start_idx + self.dim, 0] = next_best_est_tar_pos[jTarget, :,  iStage].T
                running_params[start_idx + self.dim: start_idx + (self.dim + self.dim), 0] = \
                    np.diag(next_best_est_tar_cov[jTarget, :, :, iStage])
                
           

            ################### part 2: robot state and command ###################
            new_idx = idx + self.nTarget * self.trac_prob.Lrtargets
            ##### for each robot, update the "known zone flag"
            ##### if robot knows the zone, then flag = 1
            next_idx = new_idx
            if com_attack_idx != -1:

                new_start_idx = new_idx
                #print("new_start_idx = ", new_start_idx)

                for jZone in range(self.danger_zones.nTypeI):
                    start_idx = new_start_idx + jZone
                    flag = self.known_typeI_flags[com_attack_idx, jZone]
                    running_params[start_idx: start_idx + 1, 0] = flag

                for jZone in range(self.danger_zones.nTypeII):
                    start_idx = new_start_idx + self.danger_zones.nTypeI + jZone
                    flag = self.known_typeII_flags[com_attack_idx, jZone]
                    running_params[start_idx: start_idx + 1, 0] = flag

                start_idx  = new_start_idx + self.danger_zones.nTypeI + self.danger_zones.nTypeII
                #print("start_idx = ", start_idx)
                running_params[start_idx: start_idx + 1, 0] = self.attacked_typeI_flags[com_attack_idx]
                running_params[start_idx + 1: start_idx + 2, 0] = self.attacked_typeII_flags[com_attack_idx]
                
                next_idx = start_idx + 2

            else:
                for iRobot in range(self.nRobot):
                    #print("`````````` iRobot = ", iRobot)
                    for jZone in range(self.danger_zones.nTypeI):
                        start_idx = new_idx + iRobot * self.danger_zones.nTypeI + jZone
                        #print("start_idx_inTypeI = ", start_idx)
                        flag = self.known_typeI_flags[iRobot, jZone]
                        running_params[start_idx: start_idx+ 1, 0] = flag
                        #print("robot {} knows the typeI zone {} flag = {}".format(iRobot, jZone, flag))
                
                # input("after iZone")
                new_idx = idx + self.forcespro.IIZoneFlag_idx[0]

                #print("new_idx = ", new_idx)
                for iRobot in range(self.nRobot):  
                    #print("TypeII `````````` iRobot = ", iRobot)  
                    for jZone in range(self.danger_zones.nTypeII):
                        start_idx = new_idx + iRobot * self.danger_zones.nTypeII + jZone
                        flag = self.known_typeII_flags[iRobot, jZone]
                        running_params[start_idx: start_idx + 1, 0] = flag
                        #print("start_idx = ", start_idx)
                        #print("robot {} knows the typeII zone {} flag = {}".format(iRobot, jZone, flag))


                new_idx = idx + self.forcespro.IAttackFlag_idx[0]
                #print("new_idx = ", new_idx)
                

                ##### for each robot, update the attack flag
                for iRobot in range(self.nRobot):
                    start_idx = new_idx + iRobot 
                    #2print("start_idx_TypeIAttack_flag = ", start_idx)
                
                    running_params[start_idx: start_idx + 1, 0] = self.attacked_typeI_flags[iRobot]
                
                for iRobot in range(self.nRobot):
                    start_idx = new_idx + self.nRobot + iRobot
                    #print("start_idx_TypeIIAttack_flag = ", start_idx)
                    #print("attacked_typeII_flags = ", self.attacked_typeII_flags[iRobot])
                    running_params[start_idx: start_idx + 1, 0] = self.attacked_typeII_flags[iRobot]

                next_idx = idx + self.forcespro.IIAttackFlag_idx[1]
                # print("running_params_size = ", running_params[iStage * stage_num: (iStage + 1) * stage_num].T)


            ################### part 3: weights for the objective function ###################
            
            #print("self.trac_prob.weights = ", self.trac_prob.weights)
            #print("self.trac_prob.weights.shape = ", len(self.trac_prob.weights))
            #print("idx = ", idx)
            #print("running_params = ", running_params.shape)
            ##### weights for the objective function
            #print("next_idx = ", next_idx)
            #print("next_idx: next_idx + len(self.trac_prob.weights) = ", next_idx, next_idx + len(self.trac_prob.weights))
            running_params[next_idx: next_idx + len(self.trac_prob.weights), 0] = self.trac_prob.weights

            ################### part 4: assignmnet matrix ###################
            next_idx = next_idx + len(self.trac_prob.weights)
            if com_attack_idx != -1:
                start_idx = next_idx
                running_params[start_idx: start_idx + self.nTarget, 0] = self.trac_prob.assignment_matrix[com_attack_idx, :]

            else:
                for iRobot in range(self.nRobot):
                    start_idx = next_idx + iRobot * self.nTarget
                    running_params[start_idx: start_idx + self.nTarget, 0] = self.trac_prob.assignment_matrix[iRobot, :]



        #print("running_params_size = ", running_params.shape)
        problem_setup["all_parameters"] = running_params
        # for i in range(self.N):
        #     print("running_params at stage {} = ".format(i), running_params[i * stage_num: (i+1) * stage_num].T)
        #     print("size = ", running_params[i * self.forcespro.Lrvars_total: (i+1) * self.forcespro.Lrvars_total].shape)
        
    

    ## yuwei: solved the bug
    def update_init_guess(self, problem_setup):
        
        one_stage_num = self.forcespro.Ldvars_each_rob * self.nRobot
        x0i = 1e-4 * np.ones((2 * one_stage_num, 1))      # so... x0i was not used in the following code?

        #[x1, y1, vx1, vy1, slack1, slack2,        ;    x2, y2, vx2, vy2, ...]
        # xinit = np.zeros((self.nRobot * self.dim + self.zone_num, 1))
        xinit = np.zeros((self.nRobot * self.dim, 1))
        
        for iRobot in range(self.nRobot):
            #if self.attacked_typeII_flags[iRobot] != 1:
            start_idx_init = iRobot * self.dim
            xinit[start_idx_init: start_idx_init + 1 * self.dim, 0] = \
            self.trac_prob.robot_odom[iRobot, :self.dim].T  # only x, y


        for i in range(self.N):
            stage_idx = i * one_stage_num
            for iRobot in range(self.nRobot):
                start_idx_x0 = stage_idx + iRobot * self.trac_prob.Ldrobots 
                if self.attacked_typeII_flags[iRobot] != 1:

                    x0i[start_idx_x0: start_idx_x0 + 1 * self.dim, 0] = \
                    self.trac_prob.robot_odom[iRobot, :self.dim].T # x, y, vx=cmd, vy=cmd
                    #print("self.trac_prob.robot_odom = ", self.trac_prob.robot_odom[iRobot, :self.dim].T)
                
                    #### better to keep the initial guess of output ~ 0
                    # if self.last_exitflag == 1: # if can solve normally
                    #     x0i[start_idx_x0 + 1 * self.dim: start_idx_x0 + 2 * self.dim, 0] = self.cmd[iRobot, :].T
                    x0i[start_idx_x0 + 1 * self.dim: start_idx_x0 + 2 * self.dim, 0] += 0.1  # avoid NAN or INF value
            

            ### slack variables
            start_idx_x0 = stage_idx + self.nRobot * self.trac_prob.Ldrobots
            x0i[start_idx_x0: one_stage_num * (i+1), 0] = 0.1

        #print("x0 = ", x0i.T)
        #problem_setup["x0"] = x0i
        problem_setup["x0"] = x0i
        problem_setup["xinit"] = xinit
        #print("xinit = ", xinit.T)
        #self.forcespro.ineq_constraintN(x0i[one_stage_num:], problem_setup["all_parameters"])
                
        return


    def update_init_guess_single(self, problem_setup, iRobot):

        one_stage_num = self.forcespro.Ldvars_each_rob
        x0i = np.zeros((2 * one_stage_num, 1))      # so... x0i was not used in the following code? yes, not accurate
        xinit = np.zeros((self.dim, 1))

        start_idx_x0 = 0
        start_idx_init = 0
        #print("iRobot = ", iRobot)
        #print(" self.attacked_typeII_flags[iRobot] = ", self.attacked_typeII_flags[iRobot])

        x0i[start_idx_x0: start_idx_x0 + 1 * self.dim, 0] = self.trac_prob.robot_odom[iRobot, :self.dim].T # x, y, vx=cmd, vy=cmd
        x0i[start_idx_x0 + 1 * self.dim: start_idx_x0 + 2 * self.dim, 0] += 1e-5  # avoid NAN or INF value
        xinit[start_idx_init: start_idx_init + 1 * self.dim, 0] = self.trac_prob.robot_odom[iRobot, :self.dim].T  # only x, y

        #print("x0 = ", x0i.T)
        #problem_setup["x0"] = x0i
        problem_setup["xinit"] = xinit
        #print("xinit = ", xinit.T)

        return        

    def extract_output(self, output):
        """
        get the next stage output
        """
        #print("output = ", output)
        # Extract output
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        temp = np.zeros((self.forcespro.Ldvars_each_rob * self.nRobot, self.N))
        for i in range(0, self.N):
            temp[:, i] = output['x{}'.format(i + 1)]
            # print and keep 2 decimal points
            #print("temp =  ", np.round(temp[:, i], 2))
            

        first_stage = 0
        next_stage = 1
        cmd = np.zeros((self.nRobot, self.dim))
        state = np.zeros((self.nRobot, self.dim))
        
        for iRobot in range(self.nRobot):
            start_idx = iRobot * self.trac_prob.Ldrobots
            #print("start_idx = ", start_idx)
            #print("start_idx: start_idx + self.dim, ", start_idx, start_idx + self.dim)
            state[iRobot, :] = temp[start_idx: start_idx + self.dim, next_stage].T
            cmd[iRobot, :] = temp[start_idx + self.dim: start_idx + 2 * self.dim, first_stage].T
            #print("state = ", np.round(state[iRobot, :], 2), "cmd = ", np.round(cmd[iRobot, :], 2))
            
        return state, cmd
    
    

    def extract_output_single(self, output):

        # Extract output
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        temp = np.zeros((self.forcespro.Ldvars_each_rob, self.N))
        #print(output)
        for i in range(0, self.N):
            temp[:, i] = output['x{}'.format(i + 1)]
            
            # print and keep 2 decimal points
            #print("temp =  ", np.round(temp[:, i], 2))
            

        first_stage = 0
        next_stage = 1
        cmd = np.zeros((1, self.dim))
        state = np.zeros((1, self.dim))
        

        start_idx = 0
        #print("start_idx = ", start_idx)
        #print("start_idx: start_idx + self.dim, ", start_idx, start_idx + self.dim)
        state = temp[start_idx: start_idx + self.dim, next_stage].T
        cmd   = temp[start_idx + self.dim: start_idx + 2 * self.dim, first_stage].T
        #print("state = ", np.round(state, 2), "cmd = ", np.round(cmd, 2))

        return state, cmd

    def step(self):
        """
        step the simulation for one iteration at time t
        """
        print("\033[92m======================= step start: current step is {} =======================\033[0m".format(self.cur_step))
        
    
        # get flag for danger zones (type II)
        self.update_attacked_robots()
        robot_with_attack = self.get_robot_with_attack()

        ############ EKF to danger zones ############
        # get target real state(propagate) at t + 1
        # compute trace at t (robot <-> best_est_tar_pos)
        ############# the state contains all the robots
        # update running parameters(using target est k+1 info) for each iteration
        problem_setup = {}
        #R_S AND R_C
        state = np.zeros((self.nRobot, self.dim))
        cmd = np.zeros((self.nRobot, self.dim))
        cur_trace = 0




        ################################ cent solver ################################
        #### if the robot attacked by type II zone is more than 1, then use centralized solver
        if np.sum(self.attacked_typeII_flags) <= self.nRobot - 1 \
            and self.use_cent_solver == True:
            #print("\033[93m---------- centralized solver \033[0m")


            cur_trace += self.trac_prob.get_trace(self.state, robot_with_attack)
            # ekf update the best_est_tar_pos at time k+1
            self.trac_prob.ekf_update(self.state, robot_with_attack)


            self.update_running_params(problem_setup, -1)
            self.update_init_guess(problem_setup)
            # solve(objN(using target est k+1 info and decision var z to get trace at time t+1),
            # output includes next state of robot) and record data
            #print("problem_setup = ", problem_setup)
            output, exitflag, info = self.cent_solver.solve(problem_setup)
            state, cmd = self.extract_output(output)
            if exitflag != 1  and exitflag != 0:
                print("not solved")
                #print(output)
                state, cmd = self.state, self.cmd
                
            #assert exitflag == 1 or exitflag == 0, "bad exitflag"
            #print(" info.pobj = ", info.pobj)
            #print("[Cent] exitflag = ", exitflag)
            self.last_exitflag = exitflag

            # -7 : The solver could not proceed. Most likely cause is that the problem is infeasible.
            # 0  : (for binary branch-and-bound) maximum computation time of codeoptions.mip.timeout reached. 
            
            # sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
            #                 .format(info.it, info.solvetime))

            # update robot real state at t+1
            #self.forcespro.ineq_constraintN(output['x{}'.format(2)], problem_setup["all_parameters"])
            #self.forcespro.ineq_constraint(output['x{}'.format(1)], problem_setup["all_parameters"])
            # self.forcespro.obj(output['x{}'.format(1)], problem_setup["all_parameters"])
            #self.forcespro.objN(output['x{}'.format(2)], problem_setup["all_parameters"])

            ##use all zeros as input
            # x_zero = np.zeros((self.forcespro.Ldvars_each_rob * self.nRobot, 1))
            # self.forcespro.obj(x_zero, problem_setup["all_parameters"])
            # self.forcespro.objN(x_zero, problem_setup["all_parameters"])
            #self.forcespro.collision_avoid_constraint(output['x{}'.format(1)], problem_setup["all_parameters"])
            #self.forcespro.trace_objective_casadi(output['x{}'.format(2)], problem_setup["all_parameters"])

        for iRobot in range(self.nRobot):
            trace = 0
            if self.attacked_typeII_flags[iRobot] == 1 or self.use_cent_solver == False:

                ############# it means the robot can update the trace at time t+1 #############
                if self.attacked_typeI_flags[iRobot] == 0:
                    trace = self.trac_prob.get_trace_single(self.state[iRobot, :], iRobot)
                    # ekf update the best_est_tar_pos at time k+1

                    if self.exp == "simulation" or self.exp == "ros simulation":
                        self.trac_prob.ekf_update_single(self.state[iRobot, :], iRobot)
                    


                single_problem_setup = {}
                self.update_running_params(single_problem_setup, iRobot)
                self.update_init_guess_single(single_problem_setup, iRobot)
                print("\033[93m---------- single solver for robot {} \033[0m".format(iRobot))
                single_output, single_exitflag, info = self.single_solver[iRobot].solve(single_problem_setup)
                #print("[Single] exitflag = ", single_exitflag)
                if single_exitflag != 1:
                    print("[Single] not solved")
                    #print(output)
                # assert exitflag == 1 or exitflag == 0, "bad exitflag"
                self.single_forcespro[iRobot].objN(single_output['x{}'.format(2)], single_problem_setup["all_parameters"])
                if single_exitflag == 1 or single_exitflag == 0:
                    # sys.stderr.write("[Single] FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
                    #                 .format(info.it, info.solvetime)
                    #                 )
                    state_i, cmd_i = self.extract_output_single(single_output)
                else:
                    state_i = self.state[iRobot, :]
                    cmd_i = self.cmd[iRobot, :]
                    #find the danger zone and move in the opposite direction
                    for j in range(self.danger_zones.nTypeII):
                        if self.inside_typeII_flags[iRobot, j] == 1:
                            #move in the opposite direction = 
                            direction = self.danger_zones.typeII_mu[j] - \
                                self.trac_prob.robot_odom[iRobot, :self.dim]
                            cmd_i = -np.ones(self.dim) * direction / np.linalg.norm(direction)

                state[iRobot, :] = state_i
                cmd[iRobot, :] = cmd_i


            self.trace_list_single[iRobot].append(trace)
            cur_trace += trace

        ############# update the robot state and command #############
        self.state = state
        self.cmd = cmd
        self.trace_list.append(cur_trace)
        # print("cmd: ", cmd.T)
        # print("state: ", state.T)
        return cmd
    

    def solve_one_step(self):
        """
        solve the problem for one step
        """
        results = {"robot_pos": [], "robot_vel": [], 
                   "target_pos": [], "robot_cmd": [], 
                   "exitflag": [], 
                   "known_typeI_flags": [], "known_typeII_flags": [], 
                   "attacked_typeI_flags": [], "attacked_typeII_flags": [],
                   "trace": [], "weights": [], "assignment_matrix": []}
        self.state[:, :] = self.trac_prob.robot_odom[:, :]


        # solver
        cmd = self.step()

        if self.exp == "simulation" or self.exp == "ros simulation":
                # peihan: comment out this line for real experiment
            self.trac_prob.update_robot(cmd)
            self.trac_prob.propagate_target_pos()
            results["robot_pos"]  = self.trac_prob.robot_odom[:, :self.dim].copy()
            results["target_pos"] = self.trac_prob.target_odom[:, :self.dim].copy()
        else:
            if self.trac_prob.is_real_init() == False:
                print("[warning]: robot or target is not initialized yet")
                print("self.trac_prob.is_robot_initialized = ", self.trac_prob.is_robot_initialized)
                print("self.trac_prob.is_target_initialized = ", self.trac_prob.is_target_initialized)
                return results
        

            next_odom = self.trac_prob.get_cmd_pos(cmd)
            results["robot_pos"] = next_odom
            results["target_pos"] = self.trac_prob.targets_best_est_pos[:, :, 0].copy()


        results["robot_vel"] = self.trac_prob.robot_odom[:, self.dim:].copy()
        results["robot_cmd"] = cmd
        results["exitflag"] = self.last_exitflag
        results["known_typeI_flags"] = self.known_typeI_flags.copy()
        results["known_typeII_flags"] = self.known_typeII_flags.copy()
        results["attacked_typeI_flags"] = self.attacked_typeI_flags.copy()
        results["attacked_typeII_flags"] = self.attacked_typeII_flags.copy()
        results["trace"] = self.trace_list[-1]

        # other dynamic settings
        results["weights"] = self.trac_prob.weights
        results["assignment_matrix"] = self.trac_prob.assignment_matrix

        self.cur_step += 1

        return results
