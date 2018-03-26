# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:29:35 2017

@author: nirin
"""
# compatibility
from __future__ import division
from __future__ import print_function

# test

# Je rajoute quelques lignes de commentaires

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

import csv
import os
from enviro_escape import *
from IO import *

from utils import *


if __name__ == '__main__':
    plt.close("all")

    args = parsing()
    Lx = args.Lx
    Ly = args.Ly
    Nx = args.Nx
    Ny = args.Ny
    T = args.T
    dt = args.dt
    numero_agent = 1
    numero_porte = -1
    seuil_porte = 0.1

    # Premier exemple

    
    nombreT = int(T / dt)


    sigma = 0.1
    epsilon = 1.0

    # Murs d'exemple
    largeur_porte = 2 * 0.90
    #==============================================================================

    murs = []
    murs.append(Obstacle([[0,3],[0,11.11]]))
    murs.append(Obstacle([[0,11.11], [13.52, 11.11]]))
    murs.append(Obstacle([[13.52, 11.11], [13.52,3]]))

    murs.append(Obstacle([[13.52 - largeur_porte, 3], [largeur_porte,3]]))
    murs.append(Obstacle([[largeur_porte,4], [largeur_porte,3]]))
    murs.append(Obstacle([[13.52 - largeur_porte, 4], [13.52 - largeur_porte,3]]))


    murs.append(Obstacle([[13.52, 3 + 1], [13.52 - 0.25, 3 + 1]]))
    murs.append(Obstacle([[13.52 - 0.25, 3 + 1], [13.52 - 0.25, 3]]))
    murs.append(Obstacle([[13.52, 3], [13.52 - 0.25, 3]]))

    murs.append(Obstacle([[0, 3 + 1], [0 + 0.25, 3 + 1]]))
    murs.append(Obstacle([[0 + 0.25, 3 + 1], [0 + 0.25, 3]]))
    murs.append(Obstacle([[0, 3], [0 + 0.25, 3]]))
    


        
    tables = []
    eleves = []
    for i in range(5):
        tables = tables + generer_table([1.5 + 0.25, 2.5 + 3 + i], [1.5 + 0.25 + 10, i + 2.9 + 3])
        for j in range(16):
            eleve = Agent(np.array([2 + j * 0.6,6.1 + i]), 1., sigma, epsilon*2)
            eleves.append(eleve)
        



    obstacles = murs + tables
    #==============================================================================


    porte1 = Porte([0,1.25], [2,1.25])
    porte2 = Porte([11.5,1.25], [13.5,1.25])


    agents = eleves

        
    portes = [porte1, porte2]
    #obstacles=build_walls(Lx,Ly,portes)

    salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)

    agents_positions = list(salleTest.maj_turns(nombreT))


    i = 0
    filename = "agents_positions.csv"
    while os.path.exists(filename):
        filename = "agents_positions_" + str(i) + ".csv"
        i += 1
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator = '\n')
        for agents_n in agents_positions:
            for number, pos, t in agents_n:
                #print(number, pos, t)
                x, y = pos
                spamwriter.writerow(list(number)+ [x, y, t])
        

    figure, axe = plt.subplots(1, 1)

    results = read_csv(filename, Lx, Ly)
    print(len(results))
    ani = animate("Titre", salleTest, results, figure, axe, len(results), dt)
    ani.save(filename + ".mp4")
    plt.show()
