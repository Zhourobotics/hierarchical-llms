#! /usr/bin/env python
import numpy as np
import scipy.special


class ConfigLoader:

    def __init__(self, config):

        ############# General Settings #############
        self.frame_id       = config['frame_id']
        self.testID         = config['testID']
        self.steps          = config['steps']
        self.exp            = config['exp']

 
        ############# LLM Settings #############
        self.llm_outer_dur = config['llm_outer_dur']
        self.llm_inner_dur = config['llm_inner_dur']
        self.llm_on        = config['llm_on']


        
        zones               = config['Zones']
        problem             = config['Problem']

        ############# Problem Settings #############
        self.drone_ids      = problem["robotID_drone"]
        self.car_ids     = problem["targetID_car"]
        self.robotID        = problem['robotID']
        self.nRobot         = len(self.robotID)
        self.targetID       = problem['targetID']
        self.nTarget        = len(self.targetID)

        ### for experiment
        self.robotHeights  = problem['robotHeights']


        ### only use if in simulation
        self.targetStartPos = problem['targetStartPos'][:int(self.nTarget)]
        self.targetStartVel = problem['targetStartVel'][:int(self.nTarget)]
        self.robotStartPos  = problem['robotStartPos'][:int(self.nRobot)]
        self.robotStartVel  = problem['robotStartVel'][:int(self.nRobot)]

        

        self.dt       = problem['dt']
        self.dim      = problem['dim']
        self.N        = problem['N']
        self.x_bounds = np.array(problem['x_bounds'])
        self.u_bounds = np.array(problem['u_bounds'])
        self.weights  = problem['weights']   

        self.assignment_matrix = np.array(problem['assignment_matrix'])
        self.task_ability           = np.array(problem['task_ability'])
        self.target_dyn_type        = problem['target_dyn_type']
        self.robot_dyn_type         = problem['robot_dyn_type']
        self.min_dist               = problem['min_dist']
        self.max_dist               = problem['max_dist']
        ## parameters for the measurement of range and bearing sensors
        self.range_peak    = problem['range_sensor'][0]
        self.range_shape   = problem['range_sensor'][1]
        self.bearing_peak  = problem['bearing_sensor'][0]
        self.bearing_shape = problem['bearing_sensor'][1]
        self.solver_name   = problem['solver_name']
        self.use_cent_solver = problem['use_cent_solver']



        self.resources     = problem['resources']


        ############# Danger Zones #############
        self.nTypeI      = zones['nTypeI']  ##### sensor zones
        self.nTypeII     = zones['nTypeII'] ##### communication zones
        self.typeI_mu    = np.array(zones['typeI_mu'])
        self.typeI_cov   = zones['typeI_cov']
        self.typeII_mu   = np.array(zones['typeII_mu'])
        self.typeII_cov  = zones['typeII_cov']


        self.typeI_delta  = zones['typeI_delta']
        self.typeII_delta = zones['typeII_delta']
        self.eps1         = zones['eps1']
        self.eps2         = zones['eps2']
        self.eps2_single  = zones['eps2_single']

        self.attack_recover_eps = zones['attack_recover_eps']
        # self.attack_recover_d = scipy.special.erfinv(1 - 2 * self.attack_recover_eps) * np.sqrt(2)
        
        self.attack_dt          = zones['attack_dt']
        self.attack_mcnt        = zones['attack_mcnt']
        self.attack_seed        = zones['attack_seed']

        self.typeI_zones  = {"typeI_mu": [], "typeI_cov": []}
        self.typeII_zones = {"typeII_mu": [], "typeII_cov": []}
        for i in range(self.nTypeI):
            self.typeI_zones["typeI_mu"].append(self.typeI_mu[i])
            self.typeI_zones["typeI_cov"].append(self.typeI_cov[i])
        for i in range(self.nTypeII):
            self.typeII_zones["typeII_mu"].append(self.typeII_mu[i])
            self.typeII_zones["typeII_cov"].append(self.typeII_cov[i])
