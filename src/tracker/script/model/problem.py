import numpy as np
import casadi
import model.dynamics as dynamics
from scipy.optimize import fminbound

class Problem:

    def __init__(self, config_loader):
        """
        problem: dict, problem setup
        """
        print("config_loader.robotID ", config_loader.robotID)
        self.exp            = config_loader.exp
        self.nRobot         = config_loader.nRobot
        self.nTarget        = config_loader.nTarget
        self.targetStartPos = config_loader.targetStartPos
        self.targetStartVel = config_loader.targetStartVel
        self.robotStartPos  = config_loader.robotStartPos
        self.robotStartVel  = config_loader.robotStartVel

        self.dt       = config_loader.dt
        self.dim      = config_loader.dim
        self.N        = config_loader.N
        self.x_bounds = config_loader.x_bounds
        self.u_bounds = config_loader.u_bounds
        self.weights  = config_loader.weights

        self.assignment_matrix = config_loader.assignment_matrix
        self.target_dyn_type        = config_loader.target_dyn_type
        self.robot_dyn_type         = config_loader.robot_dyn_type
        self.min_dist               = config_loader.min_dist
        self.max_dist               = config_loader.max_dist
        ## parameters for the measurement of range and bearing sensors
        self.range_peak    = config_loader.range_peak
        self.range_shape   = config_loader.range_shape
        self.bearing_peak  = config_loader.bearing_peak
        self.bearing_shape = config_loader.bearing_shape
        self.solver_name   = config_loader.solver_name



        self.target_dyn_func = getattr(dynamics, self.target_dyn_type)
        self.robot_dyn_func  = getattr(dynamics, self.robot_dyn_type)
        if self.target_dyn_type == 'second_order_dynamics':
            self.tstate_len = self.dim*2
        elif self.target_dyn_type == 'first_order_dynamics':
            self.tstate_len = self.dim
        if self.robot_dyn_type == 'second_order_dynamics':
            self.rstate_len = self.dim*2
        elif self.robot_dyn_type == 'first_order_dynamics':
            self.rstate_len = self.dim

        # current status of robots and targets
        self.robot_odom           = np.zeros((self.nRobot, self.rstate_len))

        # target real pos at time t
        # for simulation, we have the ground truth of target position
        self.target_odom          = np.zeros((self.nTarget, self.tstate_len))
        self.target_path_N        = np.zeros((self.nTarget, self.dim, self.N))

        # for real world, we do not have the ground truth of target position
        self.target_path_N_single = []
        for i in range(self.nRobot):
            self.target_path_N_single.append(np.zeros((self.nTarget, self.dim, self.N)))



        #target real pos at time t+1, repeated N stages

        self.is_target_initialized = np.zeros(self.nTarget)
        self.is_robot_initialized  = np.zeros(self.nRobot)
 
        ### nessary dimension definition
        self.est_pos_idx = self.dim
        self.est_cov_idx = self.dim



        ###### yuwei: check this
        self.Lrtargets = self.est_pos_idx + self.est_cov_idx
        self.Lrweights = len(self.weights)

        # x y cov(00) cov(11) # len of running parameters
        # two flags( sensor attack and communication attack)
        # it is for each robot 
        self.Ldrobots = self.dim * 2


        self.pos_idx = np.array([0, self.dim])  # start and end idx
        self.vel_idx = np.array([self.dim, self.dim * 2])  # start and end idx
        


        ################### updated variables ####################

        ##  information of targets
        # best estimation of target position at time t+1, repeated N stages
        self.targets_best_est_pos = np.zeros((self.nTarget, self.dim, self.N))
        self.targets_best_est_cov = np.zeros((self.nTarget, self.dim, self.dim, self.N))

        
        #### yuwei: one robot may have multiple targets
        self.targets_best_est_pos_single = []
        self.targets_best_est_cov_single = []
        for i in range(self.nRobot):
            self.targets_best_est_pos_single.append(np.zeros((self.nTarget, self.dim, self.N)))
            self.targets_best_est_cov_single.append(np.zeros((self.nTarget, self.dim, self.dim, self.N)))
         
    

    def is_real_init(self):
        """
        check if the real world is initialized
        """

        # print("self.is_robot_initialized: ", self.is_robot_initialized)
        # print("self.is_target_initialized: ", self.is_target_initialized)
        return np.sum(self.is_robot_initialized) == self.nRobot and \
               np.sum(self.is_target_initialized) == self.nTarget




    def init_detect_target_odom(self, target_pos, target_idx, robot_idx):

        if self.is_target_initialized[target_idx] == 1:
            return

        ## initialize the target path and target estimation
        for stage in range(self.N):
            self.targets_best_est_pos[target_idx, :, stage] = target_pos
            self.targets_best_est_cov[target_idx, :, :, stage] = np.eye(self.dim) * 0.2
            self.target_path_N_single[robot_idx][target_idx, :, stage] = target_pos


        for stage in range(self.N):
            self.targets_best_est_pos_single[robot_idx][target_idx, :, stage] = target_pos
            self.targets_best_est_cov_single[robot_idx][target_idx, :, :, stage] = np.eye(self.dim) * 0.2
    
    
        self.is_target_initialized[target_idx] = 1


    
    def update_detect_target_odom(self, detect_target_pos, target_idx, robot_idx):
        """
        update the real target odom
        """
        #print("detect_target_pos: ", detect_target_pos)

        for stage in range(self.N):
            self.target_path_N_single[robot_idx][target_idx, :, stage] = detect_target_pos
    


    def update_real_robot(self, robot_pos, idx):
        """
        update the real robot odom
        """
        self.robot_odom[idx, :self.dim] = robot_pos
        self.is_robot_initialized[idx]  = 1


    ########################## simulation functions ############################
    def init_sim(self):
        """
        init problem setup parameters
        """
        ## initialize the robot and target odom
        self.robot_odom[:, :self.dim] = self.robotStartPos
        #self.robot_odom[:, self.dim:] = self.robotStartVel
        self.target_odom[:, :self.dim] = self.targetStartPos
        self.target_odom[:, self.dim:] = self.targetStartVel

        ## initialize the target path and target estimation
        for stage in range(self.N):
            self.targets_best_est_pos[:, :, stage] = self.targetStartPos
            self.targets_best_est_cov[:, :, :, stage] = np.eye(self.dim) * 0.2
            self.target_path_N[:, :, stage] = self.targetStartPos

        for i in range(self.nRobot):
            for stage in range(self.N):
                self.targets_best_est_pos_single[i][:, :, stage] = self.targetStartPos
                self.targets_best_est_cov_single[i][:, :, :, stage] = np.eye(self.dim) * 0.2

    

    def sim_next_robot_i(self, cur_odom_i, cmd):
        nextodom = self.robot_dyn_func(cmd, cur_odom_i, self.dt)
        return nextodom

    # def sim_next_target_i(self, cur_odom_i, cmd):
    #     nextodom = self.target_dyn_func(cmd, cur_odom_i, self.dt)
    #     return nextodom

    def sim_next_robot(self, cur_odom, cmd):
        """
        cur_odom: self.nRobot x self.dim * 2
        cmd : self.nRobot x self.dim
        """
        next_robot_odom = np.zeros((self.nRobot, self.rstate_len))
        for i in range(self.nRobot):
            next_robot_odom[i] = self.sim_next_robot_i(cur_odom[i], cmd[i])
        return next_robot_odom

    def sim_next_target(self, cur_odom):
        """
        cur_odom: self.nTarget x self.dim * 2
        """
        next_target_odom = np.zeros((self.nTarget, self.tstate_len))
        u_tar = np.zeros(self.dim) # constant velocity for target
        for i in range(self.nTarget):
            next_target_odom[i] = self.target_dyn_func(u_tar, cur_odom[i], self.dt)
        return next_target_odom

    def update_robot(self, cmd):
        """
        update_real_robot_odom
        cmd: self.nRobot x self.dim
        """
        self.robot_odom = self.sim_next_robot(self.robot_odom, cmd)
        return self.robot_odom
    
    
    def get_cmd_pos(self, cmd):

        next_odom = self.sim_next_robot(self.robot_odom, cmd)
        return next_odom
    



    def propagate_target_pos(self):
        """
        propagate the target real position for N stages, and update the target odom
        """
        next = self.sim_next_target(self.target_odom)
        pos_next = next[:, :self.dim]
        for stage in range(self.N):
            self.target_path_N[:, :, stage] = pos_next
        self.target_odom = next


    def cov_intersection(self, mu_a, cov_a, mu_b, cov_b):
        """
        Fuses two estimates using the Covariance Intersection algorithm.

        Args:
            mu_a: Mean of the first estimate.
            cov_a: Covariance matrix of the first estimate.
            mu_b: Mean of the second estimate.
            cov_b: Covariance matrix of the second estimate.

        Returns:
            Fused mean and covariance matrix.
        """

        def objective_function(omega):
            """
            Objective function to minimize for finding the optimal weight.
            """
            cov = np.linalg.inv(omega * np.linalg.inv(cov_a) + (1 - omega) * np.linalg.inv(cov_b))
            return np.trace(cov)

        # Find the optimal weight using fminbound
        omega_optimal = fminbound(objective_function, 0, 1)

        # Calculate the fused covariance matrix
        cov_fused = np.linalg.inv(omega_optimal * np.linalg.inv(cov_a) + (1 - omega_optimal) * np.linalg.inv(cov_b))

        # Calculate the fused mean
        mean_fused = cov_fused @ (omega_optimal * np.linalg.inv(cov_a) @ mu_a + (1 - omega_optimal) * np.linalg.inv(cov_b) @ mu_b)

        return mean_fused, cov_fused
    
    
    def merge_estimation_to_central(self, rob_idx, robot_with_attack):

        ##find target that is not in the central estimation
        
        for i in range(self.nTarget):

            for j in range(self.nRobot):
                # if robot j is under attack, not merge
                if robot_with_attack[j] == 1:
                    continue
                if self.assignment_matrix[j, i] == 0 and self.assignment_matrix[rob_idx, i] == 1:
                    #merge two estimation
                    for stage in range(self.N):
                        # get the current estimation of tar at time k
                        self.targets_best_est_pos[i, :, stage] = self.targets_best_est_pos_single[rob_idx][i, :, stage]
                        self.targets_best_est_cov[i, :, :, stage] = self.targets_best_est_cov_single[rob_idx][i, :, :, stage]


    def get_trace(self, robot_pos_t, robot_with_attack):
        """
        get trace at the current time t
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        trace = 0
        target_est_pos_t = self.targets_best_est_pos[:, :, 0]
        target_est_cov_t = self.targets_best_est_cov[:, :, :, 0]

        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            tar_est_pos_k = target_est_pos_t[tar, :].T
            tar_est_cov_k = target_est_cov_t[tar, :]

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            # for the observation by all robots
            H_k1 = []
            R_k1 = []

            for rob in range(self.nRobot):
                if self.assignment_matrix[rob, tar] == 0 or robot_with_attack[rob] == 1:
                    continue
                rob_pos_k = robot_pos_t[rob, :self.dim].T
                p_tilde = rob_pos_k - tar_est_pos_pred
                p_prod = np.dot(p_tilde, p_tilde) + 1e-4
                p_norm = np.sqrt(p_prod)
                # get ranging and bearing sensors' state vector
                h_range = -1 * p_tilde / p_norm
                h_bearing = rot @ p_tilde / p_prod
                # measurement matrix for rob at time k
                H_rob_k1 = np.array([h_range, h_bearing])
                H_k1.append(H_rob_k1)
                #H_k1[0 * self.dim: (0 + 1) * self.dim, :] = H_rob_k1

                # get measurement error and R for rob at time k
                dist_k1 = p_norm
                R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1) + 1e-4
                R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1) + 1e-4
                R_k1.append(1 / R_range_inv)
                R_k1.append(1 / R_bearing_inv)

            # measurement update for this target:
            R = np.diag(R_k1)
            H_k1 = np.array(H_k1).reshape(-1, self.dim)

            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            trace += np.trace(tar_est_cov_k1)

        return trace
    
    def get_trace_single(self, robot_pos_t, rob_idx):

        """
        get trace at the current time t
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        trace = 0
        target_est_pos_t = self.targets_best_est_pos_single[rob_idx][:, :, 0]
        target_est_cov_t = self.targets_best_est_cov_single[rob_idx][:, :, :, 0]

        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            tar_est_pos_k = target_est_pos_t[tar, :].T
            tar_est_cov_k = target_est_cov_t[tar, :]

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            # for the observation by all robots
            H_k1 = []
            R_k1 = []


            if self.assignment_matrix[rob_idx, tar] == 0:
                continue
            rob_pos_k = robot_pos_t[0 :self.dim].T
            p_tilde = rob_pos_k - tar_est_pos_pred
            p_prod = np.dot(p_tilde, p_tilde) + 1e-4
            p_norm = np.sqrt(p_prod)
            # get ranging and bearing sensors' state vector
            h_range = -1 * p_tilde / p_norm
            h_bearing = rot @ p_tilde / p_prod
            # measurement matrix for rob at time k
            H_rob_k1 = np.array([h_range, h_bearing])
            H_k1.append(H_rob_k1)
            #H_k1[0 * self.dim: (0 + 1) * self.dim, :] = H_rob_k1

            # get measurement error and R for rob at time k
            dist_k1 = p_norm
            R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1) + 1e-4
            R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1) + 1e-4
            R_k1.append(1 / R_range_inv)
            R_k1.append(1 / R_bearing_inv)

            # measurement update for this target:
            R = np.diag(R_k1)
            H_k1 = np.array(H_k1).reshape(-1, self.dim)

            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            trace += np.trace(tar_est_cov_k1)

        return trace


    def ekf_update(self, robot_pos_t, robot_with_attack):
        """
        EKF update the best estimation of targets at time t+1
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], 
                        [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            # get the current estimation of tar at time k
            tar_est_pos_k = self.targets_best_est_pos[tar, :, 1].T
            tar_est_cov_k = self.targets_best_est_cov[tar, :, :, 1].T

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            #print("tar_est_pos_pred: ", tar_est_pos_pred)
            #print("tar_est_pos_k: ", tar_est_pos_k)

            # for the observation by all robots
            H_k1 = []
            R_k1 = []
            z_tildes = []

            for rob in range(self.nRobot):
                ## skip the robot that is under attack
                if self.assignment_matrix[rob, tar] == 0 or robot_with_attack[rob] == 1:
                    continue

                # truth position for target at k+1
                if self.exp == "ros real":
                    tar_truth_pos = self.target_path_N_single[rob][tar, :, 1]
                else:
                    tar_truth_pos = self.target_path_N[tar, :, 1]
                    
                # diff position between robot and target prediction
                rob_pos_k1 = robot_pos_t[rob, :self.dim] 
                p_tilde = rob_pos_k1 - tar_est_pos_pred
                p_prod = np.dot(p_tilde, p_tilde) + 1e-4
                p_norm = np.sqrt(p_prod)
                # get ranging and bearing sensors' state vector
                h_range = -1 * p_tilde / p_norm
                h_bearing = rot @ p_tilde / p_prod

                # measurement matrix for rob at time k+1
                H_rob_k1 = np.array([h_range, h_bearing])
                H_k1.append(H_rob_k1)

                # get measurement error and R for rob at time k+1
                # generate the noise for the measurement
                dist_k1 = p_norm
                R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1)
                R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1)
                R_k1.append(1 / (R_range_inv + 1e-4))
                R_k1.append(1 / (R_bearing_inv + 1e-4))

                range_noise = np.random.normal(0, 1 / (R_range_inv + 1e-4))
                bearing_noise = np.random.normal(0, 1 / (R_bearing_inv + 1e-4))

                z_tilde_rob = H_rob_k1 @ (tar_truth_pos - tar_est_pos_pred) + np.array([range_noise, bearing_noise])
                z_tildes.append(z_tilde_rob)

            # measurement update for this target:
            R = np.diag(R_k1)
            H_k1 = np.array(H_k1).reshape(-1, self.dim)
            z_tildes = np.array(z_tildes).reshape(-1, 1)

            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_pos_k1 = tar_est_pos_pred + (K @ z_tildes).reshape(-1)
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            # update the best estimation for time k+1
            for iStage in range(self.N):
                self.targets_best_est_pos[tar, :, iStage] = tar_est_pos_k1.T
                self.targets_best_est_cov[tar, :, :, iStage] = tar_est_cov_k1



    def ekf_update_single(self, robot_pos_t, rob_idx):
        """
        EKF update the best estimation of targets at time t+1
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], 
                        [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots\
        print("self.assignment_matrix[rob_idx, :]: ", self.assignment_matrix[rob_idx, :])
        for tar in range(self.nTarget):
            # get the current estimation of tar at time k
            tar_est_pos_k = self.targets_best_est_pos_single[rob_idx][tar, :, 1].T
            tar_est_cov_k = self.targets_best_est_cov_single[rob_idx][tar, :, :, 1].T

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            # truth position for target at k+1
            if self.exp == "ros real":
                tar_truth_pos = self.target_path_N_single[rob_idx][tar, :, 1]
            else:
                tar_truth_pos = self.target_path_N[tar, :, 1]

            # for the observation by all robots
            H_k1 = []
            R_k1 = []
            z_tildes = []
  

            if self.assignment_matrix[rob_idx, tar] == 0:
                continue

            # diff position between robot and target prediction
            rob_pos_k1 = robot_pos_t[0 :self.dim]
            p_tilde = rob_pos_k1 - tar_est_pos_pred
            p_prod = np.dot(p_tilde, p_tilde) + 1e-4
            p_norm = np.sqrt(p_prod)
            # get ranging and bearing sensors' state vector
            h_range = -1 * p_tilde / p_norm
            h_bearing = rot @ p_tilde / p_prod

            # measurement matrix for rob at time k+1
            H_rob_k1 = np.array([h_range, h_bearing])
            H_k1.append(H_rob_k1)

            # get measurement error and R for rob at time k+1
            # generate the noise for the measurement
            dist_k1 = p_norm
            R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1)
            R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1)
            R_k1.append(1 / (R_range_inv + 1e-4))
            R_k1.append(1 / (R_bearing_inv + 1e-4))

            range_noise = np.random.normal(0, 1 / (R_range_inv + 1e-4))
            bearing_noise = np.random.normal(0, 1 / (R_bearing_inv + 1e-4))

            z_tilde_rob = H_rob_k1 @ (tar_truth_pos - tar_est_pos_pred) + np.array([range_noise, bearing_noise])
            z_tildes.append(z_tilde_rob)

            # measurement update for this target:
            R = np.diag(R_k1)
            H_k1 = np.array(H_k1).reshape(-1, self.dim)
            z_tildes = np.array(z_tildes).reshape(-1, 1)

            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_pos_k1 = tar_est_pos_pred + (K @ z_tildes).reshape(-1)
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            # update the best estimation for time k+1
            for iStage in range(self.N):
                self.targets_best_est_pos_single[rob_idx][tar, :, iStage] = tar_est_pos_k1.T
                self.targets_best_est_cov_single[rob_idx][tar, :, :, iStage] = tar_est_cov_k1
        

            print("robot {} is update target {} est pos: ".format(rob_idx, tar), tar_est_pos_k)
            print("robot {} is update target {} truth pos: ".format(rob_idx, tar), tar_truth_pos)





    def trace_objective_casadi(self, z, p, IAttackFlag_idx, IIAttackFlag_idx, assigns_idx):
        """
        get trace value, casadi version
        last stage objective function at each iteration
        @param:  z: decision vars: robot [x, y, vx, vy, ax, ay, robot2:...] at last stage
        @param:  p: running parameters at last stage:
        [target1 best_estimation_pos at k+1(x, y),
        target1 best_estimation_covat k+1(cov(0,0), cov(1,1)), target2 ...]
        @return: trace
        """
        trace = 0
        A = casadi.SX.eye(self.dim)
        Q = 0.05 * casadi.SX.eye(self.dim)
        rot = casadi.SX(np.array([[0, 1], [-1, 0]]))

        typeIattack_flags = p[IAttackFlag_idx[0]: IAttackFlag_idx[1]]
        typeIIattack_flags = p[IIAttackFlag_idx[0]: IIAttackFlag_idx[1]]
        assigns = p[assigns_idx[0]: assigns_idx[1]]
        print("assigns: ", assigns.T)


        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            # get the best estimation of tar at time k+1
            tar_est_pos_k = p[tar * self.Lrtargets: tar * self.Lrtargets + self.est_pos_idx]
            tar_est_cov_k = casadi.diag(p[tar * self.Lrtargets + self.est_pos_idx: (tar + 1) * self.Lrtargets])
            print("target {} est pos: ".format(tar), tar_est_pos_k.T)
            # prediction step of EKF:
            tar_est_pos_pred = casadi.mtimes(A, tar_est_pos_k)
            tar_est_cov_pred = casadi.mtimes(casadi.mtimes(A, tar_est_cov_k),  A.T) + Q

            # for the observation by all robots
            H_k1 = []
            R_k1 = []

            for rob in range(self.nRobot):

                is_assigned = assigns[rob * self.nTarget + tar]

                #print("robot {} pos: ".format(rob), z[rob * self.Ldrobots: rob * self.Ldrobots + self.dim].T)
                
                print("robot {} is assigned to target {}? ".format(rob, tar), is_assigned)
                ### if the robot is in the type I danger zone, we do not use its observation
                ### if the robot is in the type II danger zone, the observation is not reliable
                flag = typeIIattack_flags[rob] * typeIattack_flags[rob]

                rob_pos_k1 = z[rob * self.Ldrobots: rob * self.Ldrobots + self.dim]
                p_tilde = rob_pos_k1 - tar_est_pos_pred

                # get ranging and bearing sensors' state vector
                p_prod = casadi.mtimes(p_tilde.T, p_tilde) + 1e-4
                p_norm = casadi.sqrt(p_prod)
                #print("p_norm: ", p_norm)
                #print("p_prod: ", p_prod)
                h_range = -1 * p_tilde / p_norm
                h_bearing = casadi.mtimes(rot, p_tilde) / p_prod

                # measurement matrix for rob at time k+1
                H_rob_k1 = casadi.horzcat(h_range, h_bearing)
                H_k1.append((1.0 - 1.0 * flag) * is_assigned * H_rob_k1)
                # get measurement error and R for rob at time k+1
                # generate the noise for the measurement
                dist_k1 = p_norm
                R_range_inv = self.range_peak * casadi.exp(-self.range_shape * dist_k1) + 1e-4
                R_bearing_inv = self.bearing_peak * casadi.exp(-self.bearing_shape * dist_k1) + 1e-4
                #print("R_range_inv: ", R_range_inv)
                #print("R_bearing_inv: ", R_bearing_inv)
                R_k1.append( (1.0 - 1.0 * flag) * is_assigned / R_range_inv)
                R_k1.append( (1.0 - 1.0 * flag) * is_assigned / R_bearing_inv)

            # measurement update for this target:
            R_k1 = casadi.vertcat(*R_k1)
            R    = casadi.diag(R_k1)
            H_k1 = casadi.vertcat(*H_k1)

            #print("H_k1: ", H_k1)
            #print("R: ", R)

            S = casadi.mtimes(casadi.mtimes(H_k1, tar_est_cov_pred), H_k1.T) + R
            S += casadi.SX.eye(S.shape[0]) * 1e-4
            #print("S: ", S)
            K = casadi.mtimes(casadi.mtimes(tar_est_cov_pred, H_k1.T), casadi.inv(S + 1e-4))

            tar_est_cov_k1 = tar_est_cov_pred - casadi.mtimes(casadi.mtimes(K, S), K.T)

            trace += casadi.trace(tar_est_cov_k1)


        return trace



    def trace_objective_casadi_single(self, z, p, rob_idx, assigns_idx):
        """
        get trace value, casadi version
        last stage objective function at each iteration
        @param:  z: decision vars: robot [x, y, vx, vy, ax, ay, robot2:...] at last stage
        @param:  p: running parameters at last stage:
        [target1 best_estimation_pos at k+1(x, y),
        target1 best_estimation_covat k+1(cov(0,0), cov(1,1)), target2 ...]
        @return: trace
        """
        trace = 0
        A = casadi.SX.eye(self.dim)
        Q = 0.05 * casadi.SX.eye(self.dim)
        rot = casadi.SX(np.array([[0, 1], 
                                  [-1, 0]]))
        assigns = p[assigns_idx[0]: assigns_idx[1]]


        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            # get the best estimation of tar at time k+1
            tar_est_pos_k = p[tar * self.Lrtargets: tar * self.Lrtargets + self.est_pos_idx]
            tar_est_cov_k = casadi.diag(p[tar * self.Lrtargets + self.est_pos_idx: (tar + 1) * self.Lrtargets])

            # prediction step of EKF:
            tar_est_pos_pred = casadi.mtimes(A, tar_est_pos_k)
            tar_est_cov_pred = casadi.mtimes(casadi.mtimes(A, tar_est_cov_k),  A.T) + Q

            # for the observation by all robots
            H_k1 = []
            R_k1 = []

            is_assigned = assigns[tar]

                   
            print("robot {} is assigned to target {}? ".format(rob_idx, tar), is_assigned)
            rob_pos_k1 = z[0: self.dim]
            p_tilde = rob_pos_k1 - tar_est_pos_pred
            p_prod = casadi.mtimes(p_tilde.T, p_tilde) + 1e-4
            p_norm = casadi.sqrt(p_prod)
            # get ranging and bearing sensors' state vector
            h_range = -1 * p_tilde / p_norm
            h_bearing = casadi.mtimes(rot, p_tilde) / p_prod
            # measurement matrix for rob at time k+1
            H_rob_k1 = casadi.horzcat(h_range, h_bearing)
            H_k1.append(H_rob_k1)
            # get measurement error and R for rob at time k+1
            # generate the noise for the measurement
            dist_k1 = p_norm
            R_range_inv = self.range_peak * casadi.exp(-self.range_shape * dist_k1)
            R_bearing_inv = self.bearing_peak * casadi.exp(-self.bearing_shape * dist_k1)

            R_k1.append(1.0 / (R_range_inv + 1e-4))
            R_k1.append(1.0 / (R_bearing_inv + 1e-4))

            # measurement update for this target:
            R_k1 = casadi.vertcat(*R_k1)
            R = casadi.diag(R_k1)
            H_k1 = casadi.vertcat(*H_k1)

            S = casadi.mtimes(casadi.mtimes(H_k1, tar_est_cov_pred), H_k1.T) + R
            K = casadi.mtimes(casadi.mtimes(tar_est_cov_pred, H_k1.T), casadi.inv(S))

            tar_est_cov_k1 = tar_est_cov_pred - casadi.mtimes(casadi.mtimes(K, S), K.T)

            trace += is_assigned * casadi.trace(tar_est_cov_k1)


        return trace


