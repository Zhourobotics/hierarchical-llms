import numpy as np
import scipy
import casadi
import scipy.integrate as integrate
from scipy.stats import norm
import time


class DangerZones:

    def __init__(self, config_loader):

        #### danger zones information ####
        self.nTypeI       = config_loader.nTypeI
        self.nTypeII      = config_loader.nTypeII
        self.typeI_mu     = config_loader.typeI_mu
        self.typeI_cov    = config_loader.typeI_cov
        self.typeII_mu    = config_loader.typeII_mu
        self.typeII_cov   = config_loader.typeII_cov
        self.typeI_delta  = config_loader.typeI_delta
        self.typeII_delta = config_loader.typeII_delta
        self.eps1         = config_loader.eps1
        self.eps2         = config_loader.eps2
        self.eps2_single  = config_loader.eps2_single

        #### attack model ####
        self.attack_mcnt = config_loader.attack_mcnt
        self.attack_dt = config_loader.attack_dt
        np.random.seed(config_loader.attack_seed)
        self.attack_recover_eps = config_loader.attack_recover_eps




    def recover_model(self, robot_pos, zone_idx, type):
        ###### we define that if the robot is not in the danger zone, it will recover ######
        if type == "typeI":
            value = self.gaussian_pdf_2d(robot_pos, self.typeI_mu[zone_idx], self.typeI_cov[zone_idx])
            if value > self.attack_recover_eps:
                return True
            else:
                return False
        
        if type == "typeII":
            value = self.gaussian_pdf_2d(robot_pos, self.typeII_mu[zone_idx], self.typeII_cov[zone_idx])
            if value > self.attack_recover_eps:
                return True
            else:
                return False
            

    def recover_comm_model(self, robot_pos, all_other_pos):
        
        for i in range(self.nTypeII):
            if self.jamming_comm_model(robot_pos, all_other_pos, i):
                print("recover comm model is triggered")
                return False
            
        return True
            

    def recover_model_w_inside(self, inside_flags):

        if inside_flags[:].sum() == 0:
            return True
        else:
            return False


    
    def jamming_comm_model(self, robot_pos, all_other_pos, zone_idx):

        # check if the robot distance 
        numer = 1e-4
        max_neighbor_dist_2 = numer
        for jRobot in range(len(all_other_pos)):
            diff = robot_pos - all_other_pos[jRobot]
            neighbor_dist = np.linalg.norm(diff) + numer
            max_neighbor_dist_2 = max(max_neighbor_dist_2, neighbor_dist)
        
        dist_to_zone = np.linalg.norm(robot_pos - self.typeII_mu[zone_idx]) + numer

        if dist_to_zone < self.typeII_delta[zone_idx] * np.sqrt(max_neighbor_dist_2):
            print("jamming comm model is triggered")
            print("max_neighbor_dist_2 = ", max_neighbor_dist_2)
            print("dist_to_zone = ", dist_to_zone)

            return True
        
        return False

    # Define the main function that computes the integral
    def compute_probability_2d(self, pdf, x, mean, std, r_i=0.1):
        
        # Define the integrand that includes the 2D PDF function
        def integrand(dx, dy):
            # Calculate the coordinates of the current point in the shifted frame
            point = np.array(x)
        
            return pdf(point, mean, std)
        
        # Perform the integration over a 2D circular region where ||[dx, dy]|| <= r_i
        ## account the computation time
        t1 = time.time()
        result, error = integrate.dblquad(
            integrand,
            -r_i, r_i,  # dx-axis integration limits
            lambda dx: -np.sqrt(r_i**2 - dx**2),  # dy-axis lower bound (within the circle)
            lambda dx: np.sqrt(r_i**2 - dx**2),   # dy-axis upper bound (within the circle)
            epsabs=1e-5, epsrel=1e-5
        )
        t2 = time.time()
        return result
    
    def gaussian_pdf_2d(self, point, mean, cov):

        cov_matrix = np.diag(cov)

        diff = point - mean
        det_cov = np.linalg.det(cov_matrix)
        inv_cov = np.linalg.inv(cov_matrix)
        exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))

        norm_factor = 1 / (2 * np.pi * np.sqrt(det_cov))

        # Compute the 2D Gaussian PDF
        pdf_value = norm_factor * np.exp(exponent)

        return pdf_value



    # attack model
    def attack_model(self, robot_pos, zone_idx, type, dt):
        """
        simulate attack
        """
        mean = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        if type == "typeI":
            mean = self.typeI_mu[zone_idx]
            cov  = self.typeI_cov[zone_idx]
            
        elif type == "typeII":
            mean = self.typeII_mu[zone_idx]
            cov  = self.typeII_cov[zone_idx]
        #gaussian distribution
        x = robot_pos

        # pdf = norm.pdf(x, mean, std_dev)
        # attack_prob = self.dt * np.max(pdf)

        # expenential attack model

        # two dimension gaussian distribution
        # attack_prob = 1 / (2 * np.pi * np.sqrt(sigma[0, 0] * sigma[1, 1])) * \
        #               np.exp(-0.5 * np.dot((x - mean).T, np.linalg.inv(sigma)).dot(x - mean))
        #pdf = norm.pdf(x, mean, cov)
        #attack_prob = self.compute_probability_2d(self.gaussian_pdf_2d, x, mean, std_dev)
        
        
        attack_prob = self.gaussian_pdf_2d(x, mean, cov)


        # print("attack_prob = ", attack_prob)
        attack_prob = attack_prob * (dt / self.attack_dt)
        attack = np.random.rand()
        #print("attack_prob is %.5f, attack is %.5f, is attacked: %s" % (attack_prob, attack, attack < attack_prob))
        
        return attack < attack_prob
    


    """
    @description: check if the robot is in the type I (sensing) danger zone
    """
    def if_in_typeI_zones(self, robot_pos):
        nRobot = len(robot_pos)
        flags = np.zeros((nRobot, self.nTypeI))

        for iRobot in range(nRobot):
            for iZone in range(self.nTypeI):
                drone_dist = np.linalg.norm(robot_pos[iRobot] - self.typeI_mu[iZone])
                #print(f"drone {iRobot} distance to zone {iZone}: {drone_dist}")
                value = self.gaussian_pdf_2d(robot_pos[iRobot], 
                                             self.typeI_mu[iZone], 
                                             self.typeI_cov[iZone])
                #print("for robot ", iRobot, " in typeI zone ", iZone, " value is ", value)
                #### if the robot is in the danger zone, the flag is 1
                if value < self.attack_recover_eps:
                    flags[iRobot][iZone] = 0
                    #print(f"drone {iRobot} is not in zone {iZone}")
                else:
                    flags[iRobot][iZone] = 1
                    #print(f"drone {iRobot} is not in zone {iZone}")
        
        return flags
    

    """
    @description: check if the robot is in the type II (communication) danger zone
    """
    def if_in_typeII_zones(self, robot_pos):
        nRobot = len(robot_pos)
        flags = np.zeros((nRobot, self.nTypeII))

        for iRobot in range(len(robot_pos)):
            for iZone in range(self.nTypeII):
                #print(f"drone {iRobot} distance to zone {iZone}: {drone_dist}")
                value = self.gaussian_pdf_2d(robot_pos[iRobot],
                                                self.typeII_mu[iZone],
                                                self.typeII_cov[iZone])
                #print("for robot ", iRobot, " in typeII zone ", iZone, " value is ", value)
                if value < self.attack_recover_eps:
                    flags[iRobot][iZone] = 0
                else:
                    flags[iRobot][iZone] = 1
                    #print(f"drone {iRobot} is not in zone {iZone}")
        
        return flags

    '''
    @description: chance constraint for type I danger zones
    '''
    def typeI_zone_value_i_casadi(self, robot_pos, zone_idx):
        """
        robot_pos: decision var [x, y] for one robot
        """
        typeI_cov_matrix = np.diag(self.typeI_cov[zone_idx])
        ai = self.typeI_mu[zone_idx] - robot_pos
        ai_norm = casadi.sqrt(casadi.mtimes(ai.T, ai) + 1e-3)
        ai_normalized = ai / ai_norm
        erf_value = scipy.special.erfinv(1 - 2 * (self.eps1))  # PL: added slack var
        sqr_term = 2 * casadi.mtimes(ai_normalized.T, casadi.mtimes(typeI_cov_matrix, ai_normalized)) + 1e-3
        cons = casadi.mtimes(ai_normalized.T, ai) - \
               erf_value * casadi.sqrt(sqr_term) - self.typeI_delta[zone_idx]
        # >= 0 is ok
        value = cons   # incoorporate slack var in the ineq constraint

        return value


    '''
    @description: chance constraint for type II danger zones
    '''
    def typeII_zone_value_i_casadi(self, robot_pos, zone_idx, all_robot_pos, num_bot, dim):
        """
        robot_i: current robot i pos decision var
        zone_idx: the index of the current zone
        all_robot_pos: the position of all robots
        num_bot: the number of robots
        """
        ## do not use norm_2 because when diff = [0, 0], it will have numerical issue
        typeII_cov_matrix = np.diag(self.typeII_cov[zone_idx])
        numer = 1e-4
        ai = self.typeII_mu[zone_idx] - robot_pos 
        #print("ai = ", ai)
        ai_norm = casadi.sqrt(casadi.mtimes(ai.T, ai) + numer)
        #print("ai_norm = ", ai_norm)
        ai_normalized = ai / ai_norm
        #print("ai_normalized = ", ai_normalized)
        erf_value = scipy.special.erfinv(2 * (self.eps2) - 1)
        sqr_term = 2 * casadi.mtimes(ai_normalized.T, casadi.mtimes(typeII_cov_matrix, ai_normalized)) + numer

        max_neighbor_dist_2 = numer
        for jRobot in range(num_bot):
            idx = jRobot * dim
            diff = robot_pos - all_robot_pos[idx:idx+dim]
            neighbor_dist = casadi.mtimes(diff.T, diff) + numer
            max_neighbor_dist_2 = casadi.fmax(max_neighbor_dist_2, neighbor_dist)
        
        # if max_neighbor_dist == -1:
        #     print("Error: max_neighbor_dist is -1")
        #print("casadi.mtimes(ai_normalized.T, ai) = ", casadi.mtimes(ai_normalized.T, ai))
        cons = casadi.mtimes(ai_normalized.T, ai) - \
               erf_value * casadi.sqrt(sqr_term) - \
               self.typeII_delta[zone_idx] * casadi.sqrt(max_neighbor_dist_2)
        
        #cons = casadi.mtimes(ai.T, ai) + slack
        print("cons = ", cons)
        value = cons
        return value
    

    def typeII_zone_value_i_casadi_single(self, robot_pos, zone_idx):
        """
        robot_pos: decision var [x, y] for one robot
        """
        typeII_cov_matrix = np.diag(self.typeII_cov[zone_idx])
        ai = self.typeII_mu[zone_idx] - robot_pos
        ai_norm = casadi.sqrt(casadi.mtimes(ai.T, ai) + 1e-4)
        ai_normalized = ai / ai_norm
        erf_value = scipy.special.erfinv(1 - 2 * (self.eps2_single))
        sqr_term = 2 * casadi.mtimes(ai_normalized.T, casadi.mtimes(typeII_cov_matrix, ai_normalized)) + 1e-4
        
        cons = casadi.mtimes(ai_normalized.T, ai) - \
               erf_value * casadi.sqrt(sqr_term) 
               
        # >= 0 is ok
        value = cons   # incoorporate slack var in the ineq constraint

        return value
