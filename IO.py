import matplotlib.pylab as plt
import matplotlib.animation as animation
import csv

def read_csv(filename, Lx, Ly):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        plt.figure()
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        xs = []
        ys = []
        colors = []
        for row in spamreader:
            
            r, g, b, x, y, t = row

            
            color = ('#%02X%02X%02X' % (int(r), int(g), int(b)))
            xs.append(float(x))
            ys.append(float(y))
            colors.append(color)
            # Print all data on a screen
        plt.scatter(xs, ys, c = colors)
    

def display(agent_positions):
    '''Displays the trajectories of all agents at all times'''
    2;

