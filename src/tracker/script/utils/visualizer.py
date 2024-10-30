import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from matplotlib.gridspec import GridSpec
import scipy.special

class Visualizer:

    def __init__(self, save_path):
        self.save_path = save_path

        # covert into year-month-day-hour-minute-second
        self.plot_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.robot_marker = ['-c', '-g', '-k']
    

    def plot_dyn(self, robot_cmd, robot_vel, size = 1):

        #only on self.fig2
        plt.figure("Robot Dynamics")  # "Robot Dynamics

        num_robots = len(robot_cmd[0])
        cmd_label = ['cmd_x', 'cmd_y', 'cmd_z']
        vel_label = ['vel_x', 'vel_y', 'vel_z']

        steps = len(robot_cmd)
        print("num_robots: ", num_robots)
        print("his_drone_vel: ", len(robot_vel))
        
        #nrows, ncols, index
        for i in range(num_robots):
            plt.subplot(num_robots, 2, 1 + i)
            t = np.arange(0, len(robot_cmd), 1)

            x = [robot_cmd[j][i][0] for j in range(len(robot_cmd))]
            y = [robot_cmd[j][i][1] for j in range(len(robot_cmd))]

            plt.plot(t, x, label = cmd_label[0])
            plt.plot(t, y, label = cmd_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('cmd')
            plt.title('robot' + str(i))
        
        for i in range(num_robots):
            plt.subplot(num_robots, 2, 1 + num_robots + i)
            t = np.arange(0, len(robot_vel), 1)
            x = [robot_vel[j][i][0] for j in range(len(robot_vel))]
            y = [robot_vel[j][i][1] for j in range(len(robot_vel))]
            
            plt.plot(t, x, label = vel_label[0])
            plt.plot(t, y, label = vel_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('vel')

        plt.savefig(self.save_path + 'dyn_cmd_' + self.plot_name  + '.png')

    def plot_pts(self, pts, color = 'cyan'):
        plt.figure("Figure 1: Simulation")


        for i in range(len(pts)):
            plt.plot(pts[i][0], pts[i][1], '+', color = color, markersize = 8)
        return
    
    
    def plot_cmd(self, robot_cmd, size = 1):

        #only on self.fig2
        plt.figure("Figure 2: Robot Commands")  #

        #use a new figure
        print ("Visualizing robot command")
        ##plot cmd with time
        # use subplots to plot 3d position
        num_robots = len(robot_cmd[0])
        vis_label = ['cmd_x', 'cmd_y', 'cmd_z']


        for i in range(num_robots):
            plt.subplot(num_robots, 1, 1 + i)
            t = np.arange(0, len(robot_cmd), 1)
            x = [robot_cmd[j][i][0] for j in range(len(robot_cmd))]
            y = [robot_cmd[j][i][1] for j in range(len(robot_cmd))]
            
            plt.plot(t, x, label = vis_label[0])
            plt.plot(t, y, label = vis_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('cmd')
            plt.title('robot' + str(i))
        
        #fig2 = plt.figure(2)
        #fig is title: vel
        plt.savefig(self.save_path + 'cmd_' + self.plot_name  + '.png')
    

    def plot_vel(self, robot_vel, size = 1):

        #only on self.fig2
        plt.figure("Robot Velocity")  # "Robot Velocity
        # set size of the figure
        #plt.figure(figsize=(10, 10))

        #use a new figure
        print ("Visualizing robot velocity")
        ##plot cmd with time
        # use subplots to plot 3d position
        num_robots = len(robot_vel[0])
        vis_label = ['vel_x', 'vel_y', 'vel_z']
        #print ("num_robots: ", num_robots)

        for i in range(num_robots):
            self.fig2 = plt.subplot(num_robots, 1, 1 + i)
            t = np.arange(0, len(robot_vel), 1)
            x = [robot_vel[j][i][0] for j in range(len(robot_vel))]
            y = [robot_vel[j][i][1] for j in range(len(robot_vel))]
            
            plt.plot(t, x, label = vis_label[0])
            plt.plot(t, y, label = vis_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('vel')
            plt.title('robot' + str(i))
        
        plt.savefig(self.save_path + 'vel_' + self.plot_name  + '.png')


    def plot_gradient_ellipse(self, pos, length, width, color = 'green'):
        plt.figure("Figure 1: Simulation")

        #print ("Visualizing gradient ellipse")
        # Plot the center of the ellipse
        #print("pos", pos)
        plt.plot(pos[0], pos[1], 'o', color = 'black', markersize = 3)
        
        # Define a gradient colormap
        cmap_colors = [color, (1, 1, 1, 0)]  # White to black gradient
        cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        
        # Define the grid
        x = np.linspace(pos[0] - length, pos[0] + length, 50)
        y = np.linspace(pos[1] - width, pos[1] + width, 50)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from center
        distances = np.sqrt((X - pos[0])**2 / length**2 + (Y - pos[1])**2 / width**2)
       

        # filter X, Y, distances which outside the ellipse
        # Create a mask for points outside the ellipse
        outside_ellipse = distances > 1

        # Mask out points outside the ellipse
        X[outside_ellipse] = np.nan
        Y[outside_ellipse] = np.nan
        distances[outside_ellipse] = np.nan
            

        # Plot the ellipse with gradient color. no boundaru
        plt.scatter(X, Y, c=distances, cmap=cmap, edgecolors='none')
        

 
    ### setup
    def visualize_map(self, x_bounds):
        plt.figure("Figure 1: Simulation", figsize=(6, 6))
        plt.axis('equal')
        # has grid
        #plt.grid(True)
        ## self.fig size is large enough to contain the map
        ## plot the map, map can be 2d or 3d box
        print ("Visualizing map")
        map_dim = len(x_bounds)
        if map_dim == 2:
            plt.plot([x_bounds[0][0], x_bounds[0][1]], [x_bounds[1][0], x_bounds[1][0]], 'black')
            plt.plot([x_bounds[0][0], x_bounds[0][1]], [x_bounds[1][1], x_bounds[1][1]], 'black')
            plt.plot([x_bounds[0][0], x_bounds[0][0]], [x_bounds[1][0], x_bounds[1][1]], 'black')
            plt.plot([x_bounds[0][1], x_bounds[0][1]], [x_bounds[1][0], x_bounds[1][1]], 'black')

        plt.axis('tight')
        plt.xlabel('x')
        plt.ylabel('y')

        return


    def visualize_zones(self, typeI_zones, typeII_zones, known_typeI_flags, known_typeII_flags):
        plt.figure("Figure 1: Simulation")

        ## zones is circle with mu as center and conv as radius
        nTypeI = len(typeI_zones["typeI_mu"])
        nTypeII = len(typeII_zones["typeII_mu"])


        print ("Visualizing zones")
        typeI_mu   = typeI_zones["typeI_mu"]
        typeI_cov  = typeI_zones["typeI_cov"]
        typeII_mu  = typeII_zones["typeII_mu"]
        typeII_cov = typeII_zones["typeII_cov"]

        rd = scipy.special.erfinv(1 - 2 * 0.0001) * np.sqrt(2)
        for i in range(nTypeI):
            #plot a disk
            # plot, fill with gradient color centered at mu
            # color is red 
            if known_typeI_flags[:, i].sum() >= 1:
                color = 'red'
            else:
                color = 'grey'
            #print("typeI_cov[i][0]", typeI_cov[i][0])
            #print("typeI_cov[i][1]", typeI_cov[i][1])
            #print("typeI_mu: ", typeI_mu)
            #print("typeI_cov: ", typeI_cov)
            self.plot_gradient_ellipse(typeI_mu[i], 
                                       rd * typeI_cov[i][0], 
                                       rd * typeI_cov[i][1], color)
            #plot the name of the zone
            plt.text(typeI_mu[i][0] + 0.05 , typeI_mu[i][1] + 0.05 , 'Izone' + str(i), fontsize=12)

        
        for i in range(nTypeII):
            #plot a disk
            if known_typeII_flags[:, i].sum() >= 1:
                color = 'blue'
            else:
                color = 'grey'
            self.plot_gradient_ellipse(typeII_mu[i], 
                                       rd * typeII_cov[i][0], 
                                       rd * typeII_cov[i][1], color)
            plt.text(typeII_mu[i][0] + 0.05 , typeII_mu[i][1] + 0.05 , 'IIzone' + str(i), fontsize=12)

    def visualize_target(self, target_pos, target_ids, size = 1):
        plt.figure("Figure 1: Simulation")

        print ("Visualizing target")

        ##plot id name as "tar" + str(i) at the start position
        for i in range(len(target_pos[0])):
            plt.text(target_pos[0][i][0] + 0.05 , target_pos[0][i][1] + 0.05 , 'tar' + str(target_ids[i]), fontsize=12)

            #plot the start position
            plt.plot(target_pos[0][i][0], target_pos[0][i][1], 'o', color = 'black', markersize = 5 * size)


        dim = len(target_pos[0])
        
        for i in range(len(target_pos)):
            color_ratio = i / len(target_pos)
            color = (color_ratio, 0, 1 - color_ratio)
            #print (target_pos[i])

            for j in range(len(target_pos[i])):
                ## for each target, plot the position
                plt.plot(target_pos[i][j][0], target_pos[i][j][1], 'o', color = color, markersize = 3*size)
                ## hold on 

            
        return
    

    def visualize_robot(self, robot_pos, robot_ids, size = 1):
        plt.figure("Figure 1: Simulation")

        print ("Visualizing robot")
        ##plot id name as "rob" + str(i) at the start position
        for i in range(len(robot_pos[0])):
            plt.text(robot_pos[0][i][0] + 0.05 , robot_pos[0][i][1] + 0.05 , 'rob' + str(robot_ids[i]), fontsize=12)

            #plot the start position
            plt.plot(robot_pos[0][i][0], robot_pos[0][i][1], 'o', color = 'black', markersize = 5 * size)


        #dim = len(robot_pos[0])
        
        for i in range(len(robot_pos)):
            color_ratio = i / len(robot_pos)
            # use different color for robots and targets
            color = (1 - color_ratio, color_ratio, 0)
            #print (robot_pos[i])

            for j in range(len(robot_pos[i])):
                ## for each robot, plot the position
                marker = '*'
                if j == 2:
                    marker = 'o'
                    color = (color_ratio, 1 - color_ratio, 0)
                plt.plot(robot_pos[i][j][0], robot_pos[i][j][1], marker, color = color, markersize = 2*size)
                ## hold on 

            
        return
    
    def show(self):
        # use time as the name of the plot
        plt.figure("Figure 1: Simulation")
        plt.legend()
        print (" the plot is saved at: ", self.save_path + 'target_' + self.plot_name  + '.png')
        plt.savefig(self.save_path + 'target_' + self.plot_name  + '.png', dpi=300)
        plt.show()
        return


    def draw_ekf(self, estx, trux, esty, truy, bound):
        plt.figure("Figure 1: Simulation")
        plt.scatter(estx, esty, marker='*')
        plt.scatter(trux, truy, marker='o')
        plt.xlim(bound[0])
        plt.ylim(bound[1])


    def create_animate(self, robot_pos, target_pos, robot_ids, target_ids, size = 1):
        """
        create a video
        robot_pos: nbot x dim
        target_pos: ntar x dim
        """
        # create a video
        fig = plt.figure(2)
        gs = GridSpec(1, 1, figure=fig)

        # Plot trajectory
        ax = fig.add_subplot(gs[:, 0])
        plt.xlim([-3.0, 3.0])
        plt.ylim([-3.0, 3.0])
        for i in range(robot_pos.shape[0]):
            ax.text(robot_pos[i, 0] + 0.05, robot_pos[i, 1] + 0.05, 'rob' + str(robot_ids[i]), fontsize=12)
            ax.plot(robot_pos[i, 0], robot_pos[i, 1], self.robot_marker[i], color='black', markersize=5 * size)

        for i in range(target_pos.shape[0]):
            ax.text(target_pos[i, 0] + 0.05, target_pos[i, 1] + 0.05, 'tar' + str(target_ids[i]), fontsize=12)
            ax.plot(target_pos[i, 0], target_pos[i, 1], '-b', color='black', markersize=5 * size)

        plt.tight_layout()



    def animate(self, robot_pos, target_pos, step, total_step , size = 1):
        """
        animate the video
        robot_pos: nbot x dim x nsteps, where nstep is the current sim step
        same as target_pos
        """
        fig = plt.gcf()  # Get current figure
        ax2 = fig.axes  # Get current axes of figure 2

        for i in range(target_pos.shape[0]):
            ax2[0].get_lines().pop(-1).remove()
        for i in range(robot_pos.shape[0]):
            ax2[0].get_lines().pop(-1).remove()

        for i in range(target_pos.shape[0]):
            ax2[0].plot(target_pos[i, 0, 0:step], target_pos[i, 1, 0:step], '-b', markersize=2*size)
        for i in range(robot_pos.shape[0]):
            ax2[0].plot(robot_pos[i, 0, 0:step], robot_pos[i, 1, 0:step], self.robot_marker[i], markersize=2 * size)


        plt.pause(0.03)
        plt.draw()


    def plot_trace(self, trace_list):
        plt.figure("Figure 3: Trace", figsize=(10, 6))

        #self.fig3 = plt.subplot(3, 1, 1)
        total_step = len(trace_list)
        plt.plot(np.arange(total_step), np.array(trace_list),'-o')
        plt.title('Trace')
        plt.savefig(self.save_path + 'trace_' + self.plot_name  + '.png')

    def plot_trace_single(self, trace_list_single):

        plt.figure("Figure 4: Trace Individual", figsize=(10, 6))
        step = len(trace_list_single[0])
        nRobot = len(trace_list_single)

        #print("trace list single: ", trace_list_single)
        for i in range(nRobot):
            plt.subplot(nRobot, 1, 1 + i)
            plt.plot(np.arange(step), np.array(trace_list_single[i]), '-x', markersize = 2)
            plt.title('robot' + str(i))
        plt.savefig(self.save_path + 'trace_single_' + self.plot_name  + '.png')

    def plot_known_flags(self, his_known_typeI_flags, his_known_typeII_flags):

        plt.figure("Figure 5: Known Zones Flags", figsize=(10, 6))

        step = len(his_known_typeI_flags)

        nTypeI = his_known_typeI_flags[0].shape[1]
        nTypeII = his_known_typeII_flags[0].shape[1]
        nRobot = his_known_typeI_flags[0].shape[0]

        print("nTypeI: ", nTypeI)
        print("nTypeII: ", nTypeII)

        total_plot = (nTypeI + nTypeII) 
        

        
        for iTypeI in range(nTypeI):
            plt.subplot(total_plot, 1, 1 + iTypeI)
            for iRobot in range(nRobot):
                plt.plot(np.arange(step), np.array(his_known_typeI_flags)[:, iRobot, iTypeI], 
                         '-x', markersize = 2, label='robot' + str(iRobot))
                #print("his_known_typeI_flags: ", np.array(his_known_typeI_flags)[:, iRobot, iTypeI])
            plt.ylim([-0.5, 1.5])
            plt.legend()
            plt.ylabel('Type I: ' + str(iTypeI))
            plt.grid(True)
            # if iTypeI == 0: 
            #     plt.title('robot' + str(iRobot))
        
        for iTypeII in range(nTypeII):
            plt.subplot(total_plot, 1, 1 + nTypeI + iTypeII )
            for iRobot in range(nRobot):
                
                plt.plot(np.arange(step), np.array(his_known_typeII_flags)[:, iRobot, iTypeII], 
                         '-x', markersize = 2, label='robot' + str(iRobot))
                #print("his_known_typeII_flags: ", np.array(his_known_typeII_flags)[:, iRobot, iTypeII])
            plt.ylim([-0.5, 1.5])
            plt.grid(True)
            plt.legend()
            plt.ylabel('Type II: ' + str(iTypeII))

        #save the plot
        plt.savefig(self.save_path + 'flags_' + self.plot_name  + '.png')
 
    
    def plot_attacked_flags(self, his_attacked_typeI_flags, his_attacked_typeII_flags):

        plt.figure("Figure 6: Attacked and Exit Flags", figsize=(10, 6))

        step = len(his_attacked_typeI_flags)
        nRobot = his_attacked_typeI_flags[0].shape[0]
        total_plot = 3
        

        
        for iRobot in range(nRobot):
            plt.subplot(total_plot, 1, 2)
            plt.plot(np.arange(step), np.array(his_attacked_typeI_flags)[:, iRobot], 
                     '-x', markersize = 2, label='robot' + str(iRobot))
            plt.ylim([-0.5, 1.5])
            plt.legend()
            plt.ylabel('Type I')
            plt.grid(True)
            # if iTypeI == 0: 
            #     plt.title('robot' + str(iRobot))
        
        for iRobot in range(nRobot):
            plt.subplot(total_plot, 1, 3)
            plt.plot(np.arange(step), np.array(his_attacked_typeII_flags)[:, iRobot], 
                     '-x', markersize = 2, label='robot' + str(iRobot))
            plt.ylim([-0.5, 1.5])
            plt.grid(True)
            plt.legend()
            plt.ylabel('Type II')
        #save the plot
        plt.savefig(self.save_path + 'attacked_flags_' + self.plot_name  + '.png')


        
    def plot_exitflag(self, exitflag):
        plt.figure("Figure 6: Attacked and Exit Flags", figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(exitflag)), exitflag, '-o') #, markersize = 8)
        plt.grid(True)
        plt.ylim([-8, 5])
        plt.ylabel('Exit Flag')
        return
