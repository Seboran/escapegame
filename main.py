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
from simulation_salles import *


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
"""Modification de la simulation de l'amphi, voir le fichier de fonctions simulation_salles et le fichier texte qui donne quelques simulations possibles
(en cours d'écriture)"""

#murs = []
#murs.append(Obstacle([[0,3],[0,11.11]]))
#murs.append(Obstacle([[0,11.11], [13.52, 11.11]]))
#murs.append(Obstacle([[13.52, 11.11], [13.52,3]]))
#
#murs.append(Obstacle([[13.52 - largeur_porte, 3], [largeur_porte,3]]))
#murs.append(Obstacle([[largeur_porte,4], [largeur_porte,3]]))
#murs.append(Obstacle([[13.52 - largeur_porte, 4], [13.52 - largeur_porte,3]]))
#
#
#murs.append(Obstacle([[13.52, 3 + 1], [13.52 - 0.25, 3 + 1]]))
#murs.append(Obstacle([[13.52 - 0.25, 3 + 1], [13.52 - 0.25, 3]]))
#murs.append(Obstacle([[13.52, 3], [13.52 - 0.25, 3]]))
#
#murs.append(Obstacle([[0, 3 + 1], [0 + 0.25, 3 + 1]]))
#murs.append(Obstacle([[0 + 0.25, 3 + 1], [0 + 0.25, 3]]))
#murs.append(Obstacle([[0, 3], [0 + 0.25, 3]]))
# 
#
#
#    
#tables = []
#eleves = []
#for i in range(5):
#    tables = tables + generer_table([1.5 + 0.25, 2.5 + 3 + i], [1.5 + 0.25 + 10, i + 2.9 + 3])
#    for j in range(16):
#        eleve = Agent(np.array([2 + j * 0.6,6.1 + i]), 1., sigma, epsilon*2)
#        eleves.append(eleve)
#    
#
#
#
#obstacles = murs + tables
#==============================================================================


#porte1 = Porte([0,1.25], [2,1.25])
#porte2 = Porte([11.5,1.25], [13.5,1.25])
#
#
#agents = eleves
#
#    
#portes = [porte1, porte2]
#obstacles=build_walls(Lx,Ly,portes)


#==============================================================================
# Simulation possible de l'amphi
#==============================================================================
#variables générales utiles à la simulation
#position de la pièce de l'angle en bas à gauche dans la fenêtre d'affichage
x0 = 1.0 #décalage en x
y0 = 3.0 #dacalage en y


#variables spécifiques à l'amphi
largeur_porte = 0.80 #on met la largeur de la demi-porte
espace_x = 1.9 #esapce entre le mur de l'amphi et le bord gauche des tables 
espace_y = 2.3 #esapce entre le mur de l'amphi et le bord inférieur des tables
largeur_table = 0.3 #largueur des tables dans l'amphi
longueur_table = 9.95 #longueur des tables de l'amphi
espace_table = 0.6 #esapce entre deux tables dans l'amphi (on considère que les tables sont equidistantes)
nbr_rangees = 6 #nombre de rangées de tables dans l'amphi
espace_table_eleve = 0.19 #espace entre le bord de la table et l'élève


"""Partie à modifier pour les différentes simulations """
portes_ouvertes = [1,2,3,4] #demi-portes ouvertes numérotées de gauche à droite de 1 à 4
occupation_ligne = [0] #vecteur indiquant dans quelle ligne ligne de table on veut mettre des élèves (de 0 à nbr_rangées)
occupation_colonne = [3] #vecteur indiquant dans à quelle position d'une grande table on veut mettre des élèves (de 0 à 16)
""" """


murs_amphi = Amphi(x0,y0,largeur_porte,portes_ouvertes)
portes_amphi = murs_amphi[1]
murs_amphi = murs_amphi[0]
tables_amphi = table_amphi(x0,y0,espace_x,espace_y,nbr_rangees,largeur_table,longueur_table,espace_table)
eleves_amphi = amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne,occupation_ligne,
                     sigma,epsilon)
prof = Agent(np.array([x0 + 6.7,y0+0.6]), 1., sigma, epsilon*2, 'prof') 
eleves_amphi.append(prof)

obstacles = murs_amphi + tables_amphi

salleTest = Environnement(x0+15, y0+10, Nx, Ny, dt, obstacles, eleves_amphi, portes_amphi)



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

portes_ouvertes = [1,2]

#simulation de la salle
Gd_salle_2_portes = Salle_233_235(x0,y0,largeur_porte_Gs,portes_ouvertes) #les portes sont préenregistrées dans cette fonction
Portes_Gs = Gd_salle_2_portes[1] #on récupère les portes
Gd_salle_2_portes = Gd_salle_2_portes[0]
#ajout des tables dans la salle
Gd_salle_tables = tables_salle_233_235(x0,y0,largeur_porte_Gs,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs)
Obstacles_Gd_salle = Gd_salle_2_portes + Gd_salle_tables
#ajout des agents
espace_agent_table = 0.17 #distance entre le bord de la table et l'élève
rangs_gauche = [1,2,3,4]
rangs_droite = [1,2,3]
eleves_Gs = Salle_233_235_occupation(x0,y0,largeur_porte_Gs,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,
                                  longueur_bureau_Gs,espace_agent_table,rangs_gauche,rangs_droite,sigma,epsilon)
grande_salleTest = Environnement(14,10,Nx,Ny,dt,Obstacles_Gd_salle,eleves_Gs,Portes_Gs)
#==============================================================================


#salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)

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
