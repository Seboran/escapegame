# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:29:35 2017

@author: nirin
"""

# test

# Je rajoute quelques lignes de commentaires

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

import csv
import os
from utils import Environnement, Agent, Obstacle, Porte
from IO import *
from forces import *
from utils import *



plt.close("all")



numero_agent = 1
numero_porte = -1
seuil_porte = 0.1

# Premier exemple

Lx = 20
Ly = 11.11
Nx = 400
Ny = 400


nombreT = 20
T = 0.1
dt = T/nombreT


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

results = read_csv("agents_positions_11.csv", Lx, Ly)
print(len(results))
ani = animate("Titre", salleTest, results, figure, axe, len(results))
ani.save("agents_positions_11.mp4")
plt.show()
