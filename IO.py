# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import matplotlib.animation as animation
from collections import defaultdict
import csv
import argparse

import os

try:
    from tqdm import tqdm as progress
except:
    print("Please install tqdm for loading bar display")
    def progress(range):
        return range

def parsing():

    parser = argparse.ArgumentParser(description='Escape game') 
    """parser.add_argument('obstacles', help='name of obstacles file', type=str)
    parser.add_argument('agents', help='name of starting agents file', type=str)"""
    parser.add_argument('-T', help='durée de la simulation', type=float, default=10.)
    parser.add_argument('-dt', help='size of dt', type=float, default=0.01)
    parser.add_argument('-Lx', help='size of room Lx', type=float, default=10.)
    parser.add_argument('-Ly', help='size of room Ly', type=float, default=10.)
    parser.add_argument('-Nx', help='size of room Lx', type=int, default=200)
    parser.add_argument('-Ny', help='size of room Lx', type=int, default=200)

    return parser.parse_args()


def read_csv(filename, Lx, Ly):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        
        
        results = defaultdict(lambda : [])
        for row in spamreader:
            
            r, g, b, x, y, t = row

            color = ('#%02X%02X%02X' % (int(r), int(g), int(b)))
            results[int(t)].append([color, float(x), float(y)])
            # Print all data on a screen
        return results
    


def animate(title, salleTest, results, fig, axe, Nt, dt):
    ''' This functions creates an animation and adds all the required legends
    Due to matplotlib limitations there are some trickled down technics to
    display a legend inside the code '''
    # Updates the screen
    def update(k, progress_bar):
        xs = []
        ys = []
        colors = []
        #print(k)
        axe.clear()
        axe.set_xlim(0, salleTest.Lx)
        axe.set_ylim(0, salleTest.Ly)
        axe.set_title(str(round(k * dt, 2)))
        salleTest.afficher(fig, axe)
        for agent_data in results[k]:
            color, x, y = agent_data
            xs.append(x)
            ys.append(y)
            colors.append(color)
        #print(xs)
        axe.scatter(xs, ys, c = colors)
        progress_bar.update(k)
        
        

    progress_bar = progress(range(Nt), desc = "Export vidéo")
        
    ''' Figure initialisation '''
    
    axe.set_xlim(0, salleTest.Lx)
    axe.set_ylim(0, salleTest.Ly)
    

    
    ani = animation.FuncAnimation(fig, update, fargs = (progress_bar,), frames = Nt, interval = dt * 1000, blit = False, repeat = True)

    
    return ani

def sauvegarde(name_export, salleTest, agents_positions, dt):
    i = 0
    filename = name_export + ".csv"
    while os.path.exists("data/" + filename):
        filename =  name_export + "_" + str(i) + ".csv"
        i += 1
    with open("data/" + filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator = '\n')
        for agents_n in agents_positions:
            for number, pos, t in agents_n:
                #print(number, pos, t)
                x, y = pos
                spamwriter.writerow(list(number)+ [x, y, t])
    

    figure, axe = plt.subplots(1, 1)
    Lx = salleTest.Lx
    Ly = salleTest.Ly
    results = read_csv("data/" + filename, Lx, Ly)
    print(len(results))
    ani = animate("Titre", salleTest, results, figure, axe, len(results), dt)
    ani.save("media/" + filename[:-4] + ".mp4") # Removes 4 lasts chars
    plt.show()



