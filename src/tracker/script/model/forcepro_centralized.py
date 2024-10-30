"""
this script is for the generation of forcespro solver and model
"""
import casadi
import numpy as np
import os, shutil

import forcespro
import forcespro.nlp
import model.dynamics as dynamics


class ForcesProSolverCent:

    def __init__(self, problem, zones):
        """
        problem: the problem class
        zones: the danger zones class
        """
        self.problem = problem
        self.zones = zones

        self.x_bounds = self.problem.x_bounds
        self.u_bounds = self.problem.u_bounds
        self.N        = self.problem.N
        self.dt       = self.problem.dt
        self.nRobot   = self.problem.nRobot
        self.nTarget  = self.problem.nTarget
        self.dim      = self.problem.dim
        self.min_dist = self.problem.min_dist
        self.max_dist = self.problem.max_dist


        #z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ... (slack_a1, slack_a2) for robot 1; ... slack2, ...]
        #p = running param: [x1, y1, cov11, cov22; x2, y2, cov11, cov22, 
        # typeI_flag for robot1, typeI_flag for robot2, ... ;
        # typeII_flag for robot1, typeII_flag for robot2, ...;
        # IAttackFlag for robot1, IAttackFlag for robot2, ...;
        # IIAttackFlag for robot1, IIAttackFlag for robot2, ...]
        self.num_zone_con = self.zones.nTypeI + self.zones.nTypeII
        self.nSlackVars = self.num_zone_con  # number of slack variables for each robot



        # index for z
        self.Ldrobots = self.problem.Ldrobots  # position and velocity index for each robot
        self.Ldslacks = self.nSlackVars # slack variables for each robot
        self.Ldvars_each_rob = self.Ldrobots + self.Ldslacks #self.dim * 2 + self.nSlackVars 


        # index for p
        # ### 1. position and covarience index
        self.Lrtargets = self.problem.Lrtargets  # running param len at one stage for one tar
        # ### 2. flags for each robot
        self.Lrflags   = self.nSlackVars + 2 
        # ### 3. weights
        self.Lrweights = self.problem.Lrweights  # =weights for each robots
        # ### 4. assignment matrix for each robot
        self.Lrassigns  = self.nTarget


        # dimension defination
        self.Lrvars_total = self.problem.Lrtargets * self.nTarget +  \
                            self.Lrflags  * self.nRobot + self.problem.Lrweights + \
                            self.Lrassigns * self.nRobot
        print("Lrvars_total: ", self.Lrvars_total)
        

        ###### index for input and position ######
        self.pos_idx = self.problem.pos_idx
        self.vel_idx = self.problem.vel_idx
        self.input_idx = self.vel_idx
        self.input_dim = self.dim

        # this index is the start index of the flags for each zones
        self.IZoneFlag_idx = np.array([self.nTarget * self.Lrtargets, 
                                       self.nTarget * self.Lrtargets + self.nRobot * self.zones.nTypeI])
        #print("IZoneFlag_idx: ", self.IZoneFlag_idx)
        self.IIZoneFlag_idx = np.array([self.IZoneFlag_idx[1],
                                        self.IZoneFlag_idx[1] + self.nRobot * self.zones.nTypeII])
        #print("IIZoneFlag_idx: ", self.IIZoneFlag_idx)
        self.IAttackFlag_idx = np.array([self.IIZoneFlag_idx[1],
                                         self.IIZoneFlag_idx[1] + self.nRobot])
        self.IIAttackFlag_idx = np.array([self.IAttackFlag_idx[1],
                                        self.IAttackFlag_idx[1] + self.nRobot])
        self.weights_idx = np.array([self.IIAttackFlag_idx[1],
                                    self.IIAttackFlag_idx[1] + 4])
        self.assigns_idx = np.array([self.weights_idx[1],
                                      self.weights_idx[1] + self.nRobot * self.Lrassigns])

    
        self.solver_name_path = "../solver/" + self.problem.solver_name + \
                                "_robot" + str(self.nRobot) + "_target" + str(self.nTarget) + \
                                "_zoneI" + str(self.zones.nTypeI) + "_zoneII" + str(self.zones.nTypeII)


        # if does not exist, generate the solver
        if not os.path.exists(self.solver_name_path):
            self.generate_solver_comm()
        

        self.solver = forcespro.nlp.Solver.from_directory(self.solver_name_path)
        





    def get_pos_var(self, z):
        """
        get the position var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_pos = casadi.SX.zeros(self.dim * self.nRobot)
        for i in range(self.nRobot):
            start_idx = i * self.Ldrobots
            #print("#### start idx ####")
            #print(start_idx)
            #print("#### pos idx ####")
            #print(self.pos_idx)
            z_pos[i * self.dim: (i + 1) * self.dim] = \
                z[start_idx + self.pos_idx[0]: start_idx + self.pos_idx[1]]
        return z_pos


    def get_pos_idx(self):
        """
        get the position index list in desicion var z
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        xinitidx = []
        for i in range(self.nRobot):
            start_idx = i * self.Ldrobots
            range_idx = range(start_idx + self.pos_idx[0], start_idx + self.pos_idx[1])
            xinitidx += range_idx
        print("#### xinitidx ####")
        print(xinitidx)
        return xinitidx


    def get_input_var(self, z):
        """
        get the input var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = casadi.SX.zeros(self.input_dim * self.nRobot)
        for i in range(self.nRobot):
            start_idx = i * self.Ldrobots
            z_input[i * self.input_dim: (i + 1) * self.input_dim] = \
                z[start_idx + self.input_idx[0]: start_idx + self.input_idx[1]]
        print("#### z_input ####")
        print(z_input)
        return z_input
    

    def get_slack_varI(self, z):
        """
        get the slack var 1 for type I danger zones, ### for testing
        """
        z_slack = casadi.SX.zeros(self.zones.nTypeI * self.nRobot)
        for i in range(self.nRobot):
            start_idx = self.nRobot * self.Ldrobots + i * self.Ldslacks
            z_slack[i * self.zones.nTypeI: (i + 1) * self.zones.nTypeI] = \
                z[start_idx: start_idx + self.zones.nTypeI]
        print("#### z_slack ####")
        print(z_slack)
        return z_slack
    

    def get_slack_varII(self, z):
        """
        get the slack var 2 for type II danger zones, ### for testing
        """
        z_slack = casadi.SX.zeros(self.zones.nTypeII * self.nRobot)
        for i in range(self.nRobot):
            start_idx = self.nRobot * self.Ldrobots + i * self.Ldslacks
            z_slack[i * self.zones.nTypeII: (i + 1) * self.zones.nTypeII] = \
                z[start_idx + self.zones.nTypeI: start_idx + self.zones.nTypeI + self.zones.nTypeII]
        print("#### z_slack ####")
        print(z_slack)
        return z_slack   # change to nTypeII and index later


    def obj(self, z, p):
        """
        control input penalty
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        #z_pos_diff = self.get_dist_diff(z)

        cur_weights = p[self.IIAttackFlag_idx[1]:]
                    
        typeIIattack_flags = p[self.IIAttackFlag_idx[0]: self.IIAttackFlag_idx[1]]


        obj = 0
        for i in range(self.nRobot):
            z_input_i = z_input[i * self.input_dim: (i + 1) * self.input_dim]
            obj += casadi.sum1(z_input_i ** 2) * (1 - typeIIattack_flags[i])
        print(" the objective function at each stage: ", obj)

        return cur_weights[0] * obj
    


    def objN(self, z, p):
        """
        last stage trace at each iteration
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        p = running param: [x1, y1, cov11, cov22, x2, y2, cov11, cov22, ...]
        """
        cur_weights = p[self.weights_idx[0]: self.weights_idx[1]]  # for the objective function

        ########################## add slack var to the objective function ################
        slack_var1 = self.get_slack_varI(z)
        slack_var2 = self.get_slack_varII(z)

        typeIIattack_flags = p[self.IIAttackFlag_idx[0]: self.IIAttackFlag_idx[1]]

        
        obj =  cur_weights[1] * self.problem.trace_objective_casadi(z, p, 
                                                                    self.IAttackFlag_idx, 
                                                                    self.IIAttackFlag_idx,
                                                                    self.assigns_idx)
        

        for i in range(self.nRobot):
            slack1 = slack_var1[i * self.zones.nTypeI: (i + 1) * self.zones.nTypeI] * (1-typeIIattack_flags[i])
            slack2 = slack_var2[i * self.zones.nTypeII: (i + 1) * self.zones.nTypeII] * (1-typeIIattack_flags[i])
            obj += cur_weights[2] * casadi.sum1(slack1 ** 2) + cur_weights[3] * casadi.sum1(slack2 ** 2)

            print("#### slack1 ####")
            print(slack1)
            print("#### slack2 ####")
            print(slack2)
            print(" the objective function at the last stage: ", obj)
        

        obj += 1000 * self.collision_avoid_obj(z, p)

        print(" the objective function at the last stage: ", obj)
        return obj


    ############## sensing danger zones ################
    def chance_constraints_typeI(self, z, p):
        """
        chance constraints for type I
        z = [x1, y1, vx1, vy1, slack1, slack2, x2, y2, vx2, vy2, slack1, slack2, ...]
        p = running param: [x1, y1, cov11, cov22, ;
                            flag_for constraints, ;
                            flag_sensorattack, flag_commattack, ...]
        """
        z_pos = self.get_pos_var(z)
        slack_var1 = self.get_slack_varI(z)
        typeIknown_flags  = p[self.IZoneFlag_idx[0]: self.IZoneFlag_idx[1]]
        typeIattack_flags = p[self.IAttackFlag_idx[0]: self.IAttackFlag_idx[1]]
        # print("#### typeI p ####")
        # print(p.T)
        # print("#### typeIknown_flags vec ####")
        # print("attack flags: ", typeIattack_flags)
        # print(typeIknown_flags)
        values = casadi.SX.zeros((self.nRobot, self.zones.nTypeI))
        for iRobot in range(self.nRobot):
            for iZone in range(self.zones.nTypeI):
                flag = typeIknown_flags[iRobot * self.zones.nTypeI + iZone]
                robot_pos = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]
                slack = slack_var1[iRobot * self.zones.nTypeI + iZone]
                ################# the constraints are computed as >= 0
                values[iRobot, iZone] = flag * \
                      (self.zones.typeI_zone_value_i_casadi(robot_pos, iZone)) + slack
                
                print("#### typeI Values ####")
                print(values[iRobot, iZone])
                print("#### typeI slack ####")
                print(slack)
                print("#### typeI flag ####")
                print(flag)

        print("#### typeI Values after ####")
        print(values)
        value_vec = casadi.vec(values)   # R_c[i]
        print("#### typeI Values vec ####")
        print(value_vec)
        return value_vec
    

    def chance_constraints_typeII(self, z, p):
        """
        chance constraints for type II
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_pos = self.get_pos_var(z)
        slack_var2 = self.get_slack_varII(z)
        typeIIknown_flags = p[self.IIZoneFlag_idx[0]: self.IIZoneFlag_idx[1]]  # flags for type II zones
        typeIIattack_flags = p[self.IIAttackFlag_idx[0]: self.IIAttackFlag_idx[1]]
        print("#### typeII z_pos ####")
        print(z_pos.T)
        print("#### typeII flag vec ####")
        print(typeIIknown_flags.T)
        print("#### typeII attack flags ####")
        print(typeIIattack_flags.T)

        values = casadi.SX.zeros((self.nRobot, self.zones.nTypeII))
        print("#### typeII Values before ####")
        print(values)

        for iRobot in range(self.nRobot):
            for iZone in range(self.zones.nTypeII):
                flag  = typeIIknown_flags[iRobot * self.zones.nTypeII + iZone]
                slack = slack_var2[iRobot * self.zones.nTypeII + iZone]
                robot_pos = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]

                ### we account only when robot knows flag = 1 and typeIIattack_flags = 0

                values[iRobot, iZone] = (1-typeIIattack_flags[iRobot]) * flag * \
                    (self.zones.typeII_zone_value_i_casadi(robot_pos, \
                    iZone, z_pos, self.nRobot, self.dim)) + slack
                # values[iRobot, iZone] = (1-typeIIattack_flags[iRobot]) * flag * \
                #           self.zones.typeII_zone_value_i_casadi_single_test(robot_pos, iZone, slack)

        print("#### typeII Values after, centralized ####")
        print(values)
        value_vec = casadi.vec(values)   # R_c[i]
        print("#### typeII Values vec, centralized ####")
        print(value_vec)
        ################ chance constraints are defined to >= 0
        return value_vec
    

    def collision_avoid_obj(self, z, p):


        z_pos = self.get_pos_var(z)
        typeIIattack_flags = p[self.IIAttackFlag_idx[0]: self.IIAttackFlag_idx[1]]
        values = 0
        valid_robot = casadi.sum1(1 - typeIIattack_flags)
        print("#### valid_robot ####")
        print(valid_robot)
        for iRobot in range(self.nRobot - 1):
            pos_i = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]
            for jRobot in range(iRobot + 1, self.nRobot):
                pos_j = z_pos[jRobot * self.dim: (jRobot + 1) * self.dim]
                diff = pos_i - pos_j
                diff_norm2 = casadi.mtimes(diff.T, diff) + 1e-4
                dist_sqr = (1-typeIIattack_flags[iRobot]) * (1-typeIIattack_flags[jRobot]) * \
                           diff_norm2    # distance square, min^2 <= dist_sqr <= max^2
                values += dist_sqr
        
        #values_vec = casadi.vertcat(*values) + self.min_dist**2 / (valid_robot + 1e-4) + 1e-4
        

        # we want the distance to be in the range of [min_dist, max_dist]
        obj = values + self.min_dist**2 / (valid_robot + 1e-4) + 1e-4
        
        smooth_obj = casadi.fmin(obj - self.min_dist**2, 0) ** 2 +\
                        casadi.fmax(obj - self.max_dist**2, 0) ** 2

        return  smooth_obj + 1e-4


    def collision_avoid_constraint(self, z, p):
        """
        collision avoidance between robots, casadi version
        @param:  z: decision vars: robot [x, y, vx, vy, x1, y1, vx1, vy1 ... ]
        @param:  p: running parameters at last stage:
        @return: constraints
        """
        z_pos = self.get_pos_var(z)
        typeIIattack_flags = p[self.IIAttackFlag_idx[0]: self.IIAttackFlag_idx[1]]
        values = []
        valid_robot = casadi.sum1(1 - typeIIattack_flags)
        print("#### valid_robot ####")
        print(valid_robot)
        for iRobot in range(self.nRobot - 1):
            pos_i = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]
            for jRobot in range(iRobot + 1, self.nRobot):
                pos_j = z_pos[jRobot * self.dim: (jRobot + 1) * self.dim]
                diff = pos_i - pos_j
                diff_norm2 = casadi.mtimes(diff.T, diff) + 1e-4
                dist_sqr = (1-typeIIattack_flags[iRobot]) * (1-typeIIattack_flags[jRobot]) * \
                           diff_norm2    # distance square, min^2 <= dist_sqr <= max^2
                values.append(dist_sqr)
        #values_vec = casadi.vertcat(*values) + self.min_dist**2 / (valid_robot + 1e-4) + 1e-4
        values_vec = casadi.vertcat(*values) + self.min_dist**2 / (valid_robot + 1e-4) + 1e-4
        # print("self.min_dist**2 / (valid_robot + 1e-4) + 1e-4: ", self.min_dist**2 / (valid_robot + 1e-4) + 1e-4)
        print("#### collision avoidance values ####")
        print(values_vec)
        return values_vec
    


    def eq_constraint(self, z):
        """
        equality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        z_pos = self.get_pos_var(z)

        print("#### z_input ####")
        print(z_input)

        print("#### z_pos ####")
        print(z_pos)
        return dynamics.first_order_dynamics(z_input, z_pos, self.dt)

    def ineq_constraint(self, z, p):
        """
        inequality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., slack1, slack2]
        """
        
        #cont = 1.0 * casadi.SX.ones(self.num_zone_con * self.nRobot)
        
        cont = 1.0 * casadi.SX.ones(self.num_zone_con * self.nRobot)
                
        
        return cont
    
    def ineq_constraintN(self, z, p):
        """
        inequality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., slack1, slack2]
        """
        #  self.chance_constraints_typeII(z, p),

        cont = 1.0 * casadi.SX.ones(self.zones.nTypeII * self.nRobot)
        cont1 = 1.0 * casadi.SX.ones(self.nRobot * (self.nRobot - 1) // 2)

        all_con = casadi.vertcat(self.chance_constraints_typeI(z, p),
                                 self.chance_constraints_typeII(z, p))
        # all_con = casadi.vertcat(self.chance_constraints_typeI(z, p),
        #                          self.chance_constraints_typeII(z, p),
        #                          self.collision_avoid_constraint(z, p))
        print("#### all constraints ####")
        print(all_con)
        return all_con

                            #   self.get_slack_var1(z))
    


    def get_E_matrix(self, row, col):
        """
        get the E matrix for equality constraints, on the LHS, E is num_eq x num_vars,
        E is selection matrix, select the position states from decision vars [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        E = np.zeros((row, col))
        print("#### E matrix ####")
        print(E.shape)

        position_idx, r = 0, 0
        for i in range(self.nRobot):
            E[r, position_idx] = 1
            E[r + 1, position_idx + 1] = 1
            position_idx += self.Ldrobots
            r += self.dim
        print(E)
        return E


    """
    TODO: Add slack variables
    """
    def generate_solver_comm(self):
    
        #num_collision_avoid_con = self.nRobot * (self.nRobot - 1) // 2
        num_chance_con = self.num_zone_con * self.nRobot

        model = forcespro.nlp.SymbolicModel()
        model.N = self.N

        ## one for avoid collision
        model.nvar = self.Ldvars_each_rob * self.nRobot
        print("#### model.nvar ####")
        print(model.nvar)
        print("robot num: ", self.nRobot)

        # number of chance constraints
        #model.nh = num_chance_con + num_collision_avoid_con # number of chance constraints + slack vars
        #model.nh = num_collision_avoid_con + num_chance_con
        # number of equality constraints
        model.neq = self.dim * self.nRobot  # [x1, y1, x2, y2, ...]

        # number of parameters

        # R_c = [], R_s = []
        model.npar = self.Lrvars_total
        # [x1, y1, cov11, cov22, x2, y2, cov11', cov22' ..., g_s1, g_s2, ..., g_c1, g_c2, ...]

        # obj
        model.objective =  self.obj   # t (xt, ut) #PL: modified objective w/ slack var
        model.objectiveN = self.objN   # t+1 (x_{t+1}, u_{t+1})

        #model.objectiveN = lambda z, p: self.objN(z, p)

        model.eq = lambda z: self.eq_constraint(z)
        model.E = self.get_E_matrix(model.neq, model.nvar)

        # ineq 
        # model.ineq = lambda z: self.collision_avoid_constraint(z)
        # #model.ineq = lambda z, p: self.ineq_constraint(z, p)
    
        # print("#### model.ineq ####")
        # print(model.ineq)
        # model.hl = self.min_dist**2 * np.ones(num_collision_avoid_con)
        # model.hu = self.max_dist**2 * np.ones(num_collision_avoid_con) 
        # print("#### model.hl ####")
        # print(model.hl.shape)
        model.nh = num_chance_con
        model.ineq = lambda z, p: self.ineq_constraint(z, p)
        model.ineqN = lambda z, p: self.ineq_constraintN(z, p)
        
        model.hl = -1e-4 * np.ones(num_chance_con)
        
        ####### upper_bound
        model.hu = np.inf * np.ones(num_chance_con)
        # model.nh = num_chance_con
        # model.ineq = lambda z, p: self.ineq_constraint(z, p)
        # model.ineqN = lambda z, p: self.ineq_constraintN(z, p)
        
        # model.hl = - 0.0001 * np.ones(num_chance_con)
        
        # ####### upper_bound
        # model.hu = np.inf * np.ones(num_chance_con)


        # slack variable, x \in [x1, x2]
        
        # model.hl = np.concatenate((np.zeros(num_chance_con),
        #                       self.min_dist**2 * np.ones(num_collision_avoid_con) ))
        # model.hu = np.concatenate(( np.inf * np.ones(num_chance_con),
        #                      self.max_dist**2 * np.ones(num_collision_avoid_con) ))

        bound_l_1 = np.hstack((np.array([self.x_bounds[0, 0], self.x_bounds[1, 0]]),
                               np.array([self.u_bounds[0], self.u_bounds[0]]))).reshape(-1)

        bound_l = np.hstack((np.tile(bound_l_1, self.nRobot).reshape(-1), 
                             -1e-4 * np.ones(num_chance_con))).reshape(-1)
        #print(bound_l)
        
        ####### the slack variable bounds. If you dont want to use slack var, set the max_value to 0
        max_value_typeI  = 5 * self.x_bounds[0, 1]
        max_value_typeII = 5 * self.x_bounds[0, 1]
        bound_u_1 = np.hstack((np.array([self.x_bounds[0, 1], self.x_bounds[1, 1]]),
                               np.array([self.u_bounds[1], self.u_bounds[1]]))).reshape(-1)
        bound_u = np.hstack((np.tile(bound_u_1, self.nRobot).reshape(-1), 
                             max_value_typeI  * np.ones(self.zones.nTypeI * self.nRobot), 
                             max_value_typeII * np.ones(self.zones.nTypeII * self.nRobot))).reshape(-1)



        model.lb = bound_l
        model.ub = bound_u
        # check the length of model.nvar 



        # Initial condition idx
        model.xinitidx = self.get_pos_idx()   ##### may change performance z = [x, u]

        # solver parameters defined
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 500  # Maximum number of iterations
        codeoptions.printlevel = 1 # Use printlevel = 2 to print progress (but
        #                             not for timings)
        codeoptions.optlevel = 0  # 0 no optimization, 1 optimize for size,
        #                             2 optimize for speed, 3 optimize for size & speed
        codeoptions.overwrite = 1
        codeoptions.cleanup = False
        codeoptions.timing = 1

        codeoptions.noVariableElimination = 1
        codeoptions.nlp.TolStat = 1E-4
        codeoptions.nlp.TolEq = 1E-4
        codeoptions.nlp.TolIneq = 1E-4
        codeoptions.nlp.TolComp = 1E-4

        # Creates code for symbolic model formulation given above, then contacts
        # server to generate new solver
        solver = model.generate_solver(options=codeoptions)

        # move the generated solver to the folder
        if not os.path.exists(self.solver_name_path):
            os.makedirs(self.solver_name_path)
        
        # rename the FORCESNLPsolver folder to the folder_name

        os.rename('FORCESNLPsolver', self.solver_name_path)

        # move all the files with the last string as .forces
        for file in os.listdir('.'):
            if file.endswith('.forces'):
                shutil.move(file, self.solver_name_path)
            if file.endswith('.c'):
                shutil.move(file, self.solver_name_path)
            if file.endswith('.h'):
                shutil.move(file, self.solver_name_path)

        return solver, model