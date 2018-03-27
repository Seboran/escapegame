# -*- coding: utf-8 -*-

# compatibility
from __future__ import division
from __future__ import print_function


import time
import scipy
import random
import collections
import numpy as np


import matplotlib.pylab as plt
import matplotlib.animation as animation


try:
    from tqdm import tqdm as progress
except:
    print("Please install tqdm for loading bar display")
    def progress(range):
        return range


def meter_to_int(x, N, L):
    return int(x * N / L) # dégueulasse  

def int_to_meter(k, N, L):
    return float(k) * L / N

def random_agent(Lx, Ly, sigma, epsilon):
    return Agent(np.array([1 + 8 * random.random(), 1 + 13 * random.random()]), 1, sigma, epsilon)

def test_location(agent,zone):
    #la variable zone correspond à une liste de deux vecteurs qui permettent de former un carré (point angle en haut à gauche et point angle en bas à droite)
    #indique si l'agent est localisée dans la zone carrée
    lim_x_min = min(zone[0][0],zone[1][0])
    lim_x_max = max(zone[0][0],zone[1][0])
    lim_y_min = min(zone[0][1],zone[1][1])
    lim_y_max = max(zone[0][1],zone[1][1])    
    
    if(agent.position[0] >= lim_x_min and agent.position[0]<= lim_x_max):   
        if(agent.position[1] >= lim_y_min and agent.position[1]<= lim_y_max):
            return True
    return False
    
def agents_in_zone_count(agents,zone):
    #fonction qui permet de compter combien d'agents sont présents dans la zone parmis tous ceux de la liste
    nb_agent_in_zone = 0
    for agent in agents:
        in_zone = test_location(agent,zone)
        if(in_zone == True):
            nb_agent_in_zone += 1
    return nb_agent_in_zone    


def points_list(sommet_1, sommet_2):
    pt_list = []
    if(sommet_1[0] == sommet_2[0]):
        #les deux points sont alignés verticalement
        maxi = max(sommet_1[1],sommet_2[1])
        mini = min(sommet_1[1],sommet_2[1])
        diff = maxi - mini
        pt_list.append([sommet_1[0], mini])
        for i in range(1,diff):
            tmp = [sommet_1[0],mini + i]
            pt_list.append(tmp)
            
        pt_list.append([sommet_1[0],maxi])
        
        
    elif(sommet_1[1] == sommet_2[1]):
        #les deux points sont alignés horizontalement
        maxi = max(sommet_1[0],sommet_2[0])
        mini = min(sommet_1[0],sommet_2[0])
        diff = maxi - mini
        pt_list.append([mini, sommet_2[1]])
        for i in range(1,diff):
            tmp = [mini + i, sommet_1[1]]
            pt_list.append(tmp)
        pt_list.append([maxi, sommet_1[1]])
    return pt_list
    

def bfs(Nx, Ny, grid, start, goal):
    ''' Prend des coordonnés cases sans unités
    Ne pas mettre de floats ou de mètres'''
    queue = collections.deque([[start]])    
    
    grid_seen = np.zeros([Nx, Ny])
    
    grid_seen[start[0], start[1]] = 1
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < Nx and 0 <= y2 < Ny and grid[x2][y2] != 1 and grid_seen[x2, y2] != 1:
                queue.append(path + [[x2, y2]])
                grid_seen[x2, y2] = 1
    raise Exception
