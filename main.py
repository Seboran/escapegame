# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:29:35 2017

@author: nirin
"""

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
from utils import Environnement, Agent, Obstacle, Porte
from IO import *
from forces import *
from utils import *
from simulation_salle_233_235 import *


plt.close("all")



numero_agent = 1
numero_porte = -1
seuil_porte = 0.1

# Premier exemple

Lx = 20
Ly = 11.11
Nx = 400
Ny = 400


nombreT = 50000
T = 120
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


#==============================================================================
#Simulation de la Grande salle 233/235 (Gs)
#==============================================================================
largeur_porte_Gs = 0.8
x0 = 1.0 #donne le décalage en x par rapport au bord de la fenêtre
y0 = 3.0  #donne le décalage en y par rapport au bord de la fenêtre
espace_x_porte = 0.7 #espace entre la première porte et la première table
espace_x = 0.85 #espace qui separe dans la direction x deux tables
espace_y = 1.1 #espace qui separe dans la direction y deux tables
largeur_bureau_Gs = 0.5
longueur_bureau_Gs = 1.25
#simulation de la salle
Gd_salle_2_portes = Salle_233_235_deux_portes(x0,y0,largeur_porte_Gs,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs)
#ajout des portes
porte_Gs1 = Porte([x0+0.1,y0-1], [x0+0.1+largeur_porte_Gs,y0-1])
porte_Gs2 = Porte([x0+0.1+largeur_porte_Gs+5.8,y0-1], [x0+0.1+5.8+2*largeur_porte_Gs,y0-1])
Portes_Gs = [porte_Gs1,porte_Gs2]
#ajout des agents
espace_agent_table = 0.17 #distance entre le bord de la table et l'élève
eleves_Gs = Salle_233_235_full_occupation(x0,y0,largeur_porte_Gs,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs,espace_agent_table,sigma,epsilon)

grande_salleTest = Environnement(14,10,Nx,Ny,dt,Gd_salle_2_portes,eleves_Gs,Portes_Gs)
#==============================================================================


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
ani = animate("Titre", salleTest, results, figure, axe, len(results))
ani.save(filename + ".mp4")
plt.show()
