from numpy.lib.shape_base import _replace_zero_by_x_arrays
from rrt_star_reeds_shepp import RRTStarReedsShepp
import model_predictive_speed_and_steer_control 
import matplotlib.pyplot as plt
import numpy as np

obstacleList = []
map_width = 36
map_height = 33
show_animation = True

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]


def generate_map_frame(list1, width, height):

    for i in range(width):
        list1.append((i,0,0.1))
    
    for i in range(height):
        list1.append((0,i,0.1))     


def do_ReedsShepp(max_iter=500):
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [
        (7, 7, 0.5),(7,8,0.5),(7,9,0.5),(7,10,0.5),(8,7,0.5),(8,8,0.5),(8,9,0.5),(8,10,0.5),
        (10, 7, 0.5),(10,8,0.5),(10,9,0.5),(10,10,0.5),(11,7,0.5),(11,8,0.5),(11,9,0.5),(11,10,0.5),
        #(13, 7, 0.5),(13,8,0.5),(13,9,0.5),(13,10,0.5),(14,7,0.5),(14,8,0.5),(14,9,0.5),(14,10,0.5),
        (16, 7, 0.5),(16,8,0.5),(16,9,0.5),(16,10,0.5),(17,7,0.5),(17,8,0.5),(17,9,0.5),(17,10,0.5),
        (19, 7, 0.5),(19,8,0.5),(19,9,0.5),(19,10,0.5),(20,7,0.5),(20,8,0.5),(20,9,0.5),(20,10,0.5),
        (22, 7, 0.5),(22,8,0.5),(22,9,0.5),(22,10,0.5),(23,7,0.5),(23,8,0.5),(23,9,0.5),(23,10,0.5),
        (26, 7, 0.5),(26,8,0.5),(26,9,0.5),(26,10,0.5),(25,7,0.5),(25,8,0.5),(25,9,0.5),(25,10,0.5),
        (28, 7, 0.5),(28,8,0.5),(28,9,0.5),(28,10,0.5),(29,7,0.5),(29,8,0.5),(29,9,0.5),(29,10,0.5),

        (7, 18, 0.5),(7,19,0.5),(7,20,0.5),(7,21,0.5),(8,18,0.5),(8,19,0.5),(8,20,0.5),(8,21,0.5),
        #(10, 18, 0.5),(10,19,0.5),(10,20,0.5),(10,21,0.5),(11,18,0.5),(11,19,0.5),(11,20,0.5),(11,21,0.5),
        (13, 18, 0.5),(13,19,0.5),(13,20,0.5),(13,21,0.5),(14,18,0.5),(14,19,0.5),(14,20,0.5),(14,21,0.5),
        (16, 18, 0.5),(16,19,0.5),(16,20,0.5),(16,21,0.5),(17,18,0.5),(17,19,0.5),(17,20,0.5),(17,21,0.5),
        (19, 18, 0.5),(19,19,0.5),(19,20,0.5),(19,21,0.5),(20,18,0.5),(20,19,0.5),(20,20,0.5),(20,21,0.5),
        (22, 18, 0.5),(22,19,0.5),(22,20,0.5),(22,21,0.5),(23,18,0.5),(23,19,0.5),(23,20,0.5),(23,21,0.5),
        (26, 18, 0.5),(26,19,0.5),(26,20,0.5),(26,21,0.5),(25,18,0.5),(25,19,0.5),(25,20,0.5),(25,21,0.5),
        (28, 18, 0.5),(28,19,0.5),(28,20,0.5),(28,21,0.5),(29,18,0.5),(29,19,0.5),(29,20,0.5),(29,21,0.5),

        (7, 29, 0.5),(7,30,0.5),(7,31,0.5),(7,32,0.5),(8,29,0.5),(8,30,0.5),(8,31,0.5),(8,32,0.5),
        (10, 29, 0.5),(10,30,0.5),(10,31,0.5),(10,32,0.5),(11,29,0.5),(11,30,0.5),(11,31,0.5),(11,32,0.5),
        (13, 29, 0.5),(13,30,0.5),(13,31,0.5),(13,32,0.5),(14,29,0.5),(14,30,0.5),(14,31,0.5),(14,32,0.5),
        #(16, 29, 0.5),(16,30,0.5),(16,31,0.5),(16,32,0.5),(17,29,0.5),(17,30,0.5),(17,31,0.5),(17,32,0.5),
        (19, 29, 0.5),(19,30,0.5),(19,31,0.5),(19,32,0.5),(20,29,0.5),(20,30,0.5),(20,31,0.5),(20,32,0.5),
        (22, 29, 0.5),(22,30,0.5),(22,31,0.5),(22,32,0.5),(23,29,0.5),(23,30,0.5),(23,31,0.5),(23,32,0.5),
        (26, 29, 0.5),(26,30,0.5),(26,31,0.5),(26,32,0.5),(25,29,0.5),(25,30,0.5),(25,31,0.5),(25,32,0.5),
        (28, 29, 0.5),(28,30,0.5),(28,31,0.5),(28,32,0.5),(29,29,0.5),(29,30,0.5),(29,31,0.5),(29,32,0.5)
    ]  # [x,y,size(radius)]

    for i in range(36):
        obstacleList.append((i,-1,0.5))
        obstacleList.append((i,33,0.5))
    
    for i in range(33):
        obstacleList.append((-1,i,0.5))
        obstacleList.append((36,i,0.5))
    
    # for i in range(24):
    #     obstacleList.append((i+6,11,0.5))
    #     obstacleList.append((i+6,22,0.5))
        

    # Set Initial parameters
    start = [2.0, 1.5, np.deg2rad(90.0)]
    goal = [16.5, 28.5, np.deg2rad(90.0)]

    rrt_star_reeds_shepp = RRTStarReedsShepp(start, goal,
                                             obstacleList,
                                             [0, 35], max_iter=max_iter)
    path = rrt_star_reeds_shepp.planning(animation=show_animation)

    # Draw final path
    if path and show_animation:  # pragma: no cover
        rrt_star_reeds_shepp.draw_graph()
        plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
        plt.grid(True)
        plt.pause(0.001)
        plt.show()

    return path, obstacleList


def MPC_FollowPath(path, obslist):
    print(__file__ + " start!!")
    dl = 1.0  # course tick
    ax = []
    ay = []
    ayaw =[]
    for x in path:
        ax.append(x[0])
        ay.append(x[1])
        ayaw.append(x[2])


    cx, cy, cyaw, ck, s = model_predictive_speed_and_steer_control.cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    sp = model_predictive_speed_and_steer_control.calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = model_predictive_speed_and_steer_control.State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    cyaw[-1] =  ayaw[-1]
    t, x, y, yaw, v, d, a = model_predictive_speed_and_steer_control.do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state, obslist)

    if show_animation:  # pragma: no cover
        #plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()



def main(): 
    # Finding the path to the parking space
    path,obslist = do_ReedsShepp(max_iter=500)
    print(__file__ + " start!!")
    print("The path is: ")
    #print(path)
    dl = 1.0  # course tick
    
    # Transfer the path into the input to MPC
    print("The length of the path is: " +  str(len(path)))
    a = np.array(path)
    path2 = a[0:len(path)-1:len(path)//35]
    np.insert(path2,0,path[0],axis=None)
    path_final = path2[::-1]
    path_final[0] = path[-1]
    print("The final path is: ")
    print(path_final)
    
    # Using MPC to follow the path
    MPC_FollowPath(path_final, obslist)

if __name__ == '__main__':
    main()