"""
this script is for the generation of forcespro solver and model
"""
import casadi
import numpy as np
import os, shutil

import forcespro
import forcespro.nlp
import model.dynamics as dynamics


class ForcesProSolverSingle:

    def __init__(self, problem, zones, rob_idx):
        """
        problem: the problem class
        zones: the danger zones class
        """
        self.rob_idx = rob_idx
        self.problem = problem
        self.zones = zones

        self.x_bounds = self.problem.x_bounds
        self.u_bounds = self.problem.u_bounds
        self.N        = self.problem.N
        self.dt       = self.problem.dt
        self.nTarget  = self.problem.nTarget
        self.dim      = self.problem.dim

        self.num_zone_con = self.zones.nTypeI + self.zones.nTypeII
        self.nSlackVars = self.num_zone_con  # number of slack variables for each robot

        # index for z
        self.Ldrobots = self.problem.Ldrobots  # position and velocity index for each robot
        self.Ldslacks = self.nSlackVars 
        self.Ldvars_each_rob = (self.Ldrobots + self.Ldslacks) #self.dim * 2 + self.nSlackVars 


        # index for p
        # ### 1. position and covarience index
        self.Lrtargets = self.problem.Lrtargets  # running param len at one stage for one tar
        # ### 2. flags for each robot
        self.Lrflags   = self.nSlackVars + 2 
        # ### 3. weights
        self.Lrweights = self.problem.Lrweights  # =weights for each robots
        # ### 4. assignment matrix for each robot
        self.Lrassigns  = self.nTarget


        self.Lrvars_total = self.problem.Lrtargets * self.nTarget +  \
                            self.Lrflags + self.problem.Lrweights + self.Lrassigns
        print("Lrvars_total: ", self.Lrvars_total)


        ###### index for input and position ######
        self.pos_idx = [0, self.dim]
        self.vel_idx = [self.dim, self.dim * 2]
        self.input_idx = self.vel_idx
        self.input_dim = self.dim


        self.IZoneFlag_idx = np.array([self.nTarget * self.Lrtargets, 
                                       self.nTarget * self.Lrtargets + 1 * self.zones.nTypeI])
        print("IZoneFlag_idx: ", self.IZoneFlag_idx)
        self.IIZoneFlag_idx = np.array([self.IZoneFlag_idx[1],
                                        self.IZoneFlag_idx[1] + 1 * self.zones.nTypeII])
        print("IIZoneFlag_idx: ", self.IIZoneFlag_idx)
        self.IAttackFlag_idx = np.array([self.IIZoneFlag_idx[1],
                                         self.IIZoneFlag_idx[1] + 1])
        self.IIAttackFlag_idx = np.array([self.IAttackFlag_idx[1],
                                        self.IAttackFlag_idx[1] + 1])
        self.weights_idx = np.array([self.IIAttackFlag_idx[1],
                                    self.IIAttackFlag_idx[1] + self.Lrweights])
        self.assigns_idx = np.array([self.weights_idx[1],
                                     self.weights_idx[1] + self.Lrassigns])

        
        self.solver_name_path = "../solver/" + self.problem.solver_name + \
                                "_robot1" + "_target" + str(self.nTarget) + \
                                "_zoneI" + str(self.zones.nTypeI) + "_zoneII" + str(self.zones.nTypeII) \
                                + "_robidx" + str(self.rob_idx)


        # if does not exist, generate the solver
        if not os.path.exists(self.solver_name_path):
            self.generate_solver_comm()
        
        self.solver = forcespro.nlp.Solver.from_directory(self.solver_name_path)


    def get_pos_var(self, z):
        """
        get the position var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_pos = casadi.SX.zeros(self.dim)
        z_pos = z[self.pos_idx[0]: self.pos_idx[1]]
        return z_pos


    def get_pos_idx(self):
        """
        get the position index list in desicion var z
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        xinitidx = []
        range_idx = range(self.pos_idx[0], self.pos_idx[1])
        xinitidx += range_idx
        return xinitidx


    def get_input_var(self, z):
        """
        get the input var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = casadi.SX.zeros(self.input_dim)
        z_input = z[self.input_idx[0]: self.input_idx[1]]
        return z_input
    

    def get_slack_varI(self, z):
        """
        get the slack var 1 for type I danger zones, ### for testing
        """
        z_slack = casadi.SX.zeros(self.zones.nTypeI)
        start_idx = self.dim * 2
        z_slack = z[start_idx: start_idx + self.zones.nTypeI]
        return z_slack
    

    def get_slack_varII(self, z):
        """
        get the slack var 2 for type II danger zones, ### for testing
        """
        z_slack = casadi.SX.zeros(self.zones.nTypeII)
        start_idx = self.dim * 2 + self.zones.nTypeI
        z_slack = z[start_idx: start_idx + self.zones.nTypeII]
        return z_slack   # change to nTypeII and index later
    

    def obj(self, z, p):
        """
        control input penalty
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        cur_weights = p[self.weights_idx[0]: self.weights_idx[1]]  # for the objective function
                

        obj = cur_weights[0] * casadi.sum1(z_input ** 2)
        return obj
    


    def objN(self, z, p):
        """
        last stage trace at each iteration
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        p = running param: [x1, y1, cov11, cov22, x2, y2, cov11, cov22, ...]
        """
        cur_weights = p[self.weights_idx[0]: self.weights_idx[1]]  # for the objective function
        typeIattack_flags = p[self.IAttackFlag_idx[0]: self.IAttackFlag_idx[1]]

        ########################## add slack var to the objective function ################
        slack_var1 = self.get_slack_varI(z)
        slack_var2 = self.get_slack_varII(z)
        
        obj =  cur_weights[1] * (1.0 - typeIattack_flags) * self.problem.trace_objective_casadi_single(z, p, self.rob_idx, self.assigns_idx) + \
               cur_weights[2] * casadi.sum1(slack_var1 ** 2) + \
               cur_weights[3] * casadi.sum1(slack_var2 ** 2)
        
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
        flag_vec = p[self.IZoneFlag_idx[0]: self.IZoneFlag_idx[1]] # known flags for type I zones
        print("#### typeI p ####")
        print(p)
        print("#### typeI flag vec ####")
        print(flag_vec)
        values = casadi.SX.zeros((1, self.zones.nTypeI))
        print("#### typeI Values before ####")
        print(values)

        for iZone in range(self.zones.nTypeI):
            flag = flag_vec[iZone]
            slack = slack_var1[iZone]
            robot_pos = z_pos
            ################# the constraints are computed as >= 0
            values[0, iZone] = flag * self.zones.typeI_zone_value_i_casadi(robot_pos, iZone)\
                                 + slack

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
        flag_vec = p[self.IIZoneFlag_idx[0]: self.IIZoneFlag_idx[1]]  # flags for type II zones

        print("#### typeII p ####")
        print(p)
        print("#### typeII flag vec ####")
        print(flag_vec)

        values = casadi.SX.zeros((1, self.zones.nTypeII))
        print("#### typeII Values before ####")
        print(values)


        for iZone in range(self.zones.nTypeII):
            flag = flag_vec[iZone]
            slack = slack_var2[iZone]
            robot_pos = z_pos
            values[0, iZone] = flag * self.zones.typeII_zone_value_i_casadi_single \
                               (robot_pos, iZone) + slack

        print("#### typeII Values after ####")
        print(values)
        value_vec = casadi.vec(values)   # R_c[i]
        print("#### typeII Values vec ####")
        print(value_vec)
        ################ chance constraints are defined to >= 0
        return value_vec


    def eq_constraint(self, z):
        """
        equality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        z_pos = self.get_pos_var(z)

        return dynamics.first_order_dynamics(z_input, z_pos, self.dt)


    def ineq_constraint(self, z, p):
        """
        inequality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., slack1, slack2]
        """
        
        cont = 1.0 * casadi.SX.ones(self.num_zone_con)

        return cont

    def ineq_constraintN(self, z, p):
        """
        inequality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., slack1, slack2]
        """
        return casadi.vertcat(self.chance_constraints_typeI(z, p),
                              self.chance_constraints_typeII(z, p))

    def get_E_matrix(self, row, col):
        """
        get the E matrix for equality constraints, on the LHS, E is num_eq x num_vars,
        E is selection matrix, select the position states from decision vars [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        E = np.zeros((row, col))
        print("#### E matrix ####")
        print(E.shape)

        position_idx, r = 0, 0

        E[r, position_idx] = 1
        E[r + 1, position_idx + 1] = 1
        position_idx += self.Ldrobots
        r += self.dim
        print("#### E matrix ####")
        print(E)
        return E


    """
    TODO: Add slack variables
    """
    def generate_solver_comm(self):
    
        num_chance_con = self.num_zone_con

        model = forcespro.nlp.SymbolicModel()

        model.N = self.N
        ######## x_t+1 = f(u, x_t)
        ####### z = (x, u)
        #  n+1 state, n control input
        # number of variables 
        model.nvar = self.Ldvars_each_rob # [x1 y1 vx1 vy1, ...]  #####PL: add 1 for slack var with 1 typeI zone, not sure if I should add it here
    
        # number of chance constraints
        #model.nh = num_chance_con + num_collision_avoid_con # number of chance constraints + slack vars
        model.nh = num_chance_con
        # number of equality constraints
        model.neq = self.dim # [x1, y1, x2, y2, ...]

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


        model.ineq = lambda z, p: self.ineq_constraint(z, p)
        model.ineqN = lambda z, p: self.ineq_constraintN(z, p)

        model.hl = - 1e-4 * np.ones(num_chance_con) 
        model.hu = np.inf * np.ones(num_chance_con) 
        # slack variable, x \in [x1, x2]

        # bounds for decision variables #TODO: Bug - wrong number of bounds hl #yuwei solved
        bound_l = np.hstack((np.array([self.x_bounds[0, 0], self.x_bounds[1, 0]]),
                             np.array([self.u_bounds[0], self.u_bounds[0]]), 
                             -1e-4 * np.ones(self.num_zone_con))).reshape(-1)
        
        print("#### bound_l ####")
        bound_l = np.tile(bound_l, 1).reshape(-1)
        print(bound_l.shape)
        
        ####### the slack variable bounds. If you dont want to use slack var, set the max_value to 0
        max_value_typeI  = 5 * self.x_bounds[0, 1]
        max_value_typeII = 5 * self.x_bounds[0, 1]
        # this is represented as the upper bound of the slack variable, means the distance

        bound_u = np.hstack((np.array([self.x_bounds[0, 1], self.x_bounds[1, 1]]),
                             np.array([self.u_bounds[1], self.u_bounds[1]]), 
                             max_value_typeI * np.ones(self.zones.nTypeI),
                             max_value_typeII * np.ones(self.zones.nTypeII))).reshape(-1)
        

        bound_u = np.tile(bound_u, 1).reshape(-1)
        model.lb = bound_l
        model.ub = bound_u
        # check the length of model.nvar 



        # Initial condition idx
        model.xinitidx = self.get_pos_idx()   ##### may change performance z = [x, u]

        # solver parameters defined
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 500  # Maximum number of iterations
        codeoptions.printlevel = 1  # Use printlevel = 2 to print progress (but
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
