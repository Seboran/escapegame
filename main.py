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
from simulation_salles import *


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
    #murs.append(Obstacle([[13.52, 3 + 1], [13.52 - 0.25, 3 + 1]]))
    #murs.append(Obstacle([[13.52 - 0.25, 3 + 1], [13.52 - 0.25, 3]]))
    #murs.append(Obstacle([[13.52, 3], [13.52 - 0.25, 3]]))
    #
    #murs.append(Obstacle([[0, 3 + 1], [0 + 0.25, 3 + 1]]))
    #murs.append(Obstacle([[0 + 0.25, 3 + 1], [0 + 0.25, 3]]))
    #murs.append(Obstacle([[0, 3], [0 + 0.25, 3]]))
    #     
    #tables = []
    #eleves = []
    #for i in range(5):
    #    tables = tables + generer_table([1.5 + 0.25, 2.5 + 3 + i], [1.5 + 0.25 + 10, i + 2.9 + 3])
    #    for j in range(16):
    #        eleve = Agent(np.array([2 + j * 0.6,6.1 + i]), 1., sigma, epsilon*2)
    #        eleves.append(eleve)

    #obstacles = murs + tables
    #==============================================================================

    #porte1 = Porte([0,1.25], [2,1.25])
    #porte2 = Porte([11.5,1.25], [13.5,1.25])
    #
    #agents = eleves
    #   
    #portes = [porte1, porte2]
    #obstacles=build_walls(Lx,Ly,portes)


    #==============================================================================
    # Simulation possible de l'amphi
    #==============================================================================
    #variables générales utiles à la simulation
    #position de la pièce de l'angle en bas à gauche dans la fenêtre d'affichage
    x0 = 1.0 
    y0 = 3.0 
    largeur_porte = 0.80 #largeur d'une petite porte ou d'une demi-porte d'amphi

    #variables spécifiques à l'amphi
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
    #Simulation possible de la grande salle 233/235
    #==============================================================================
    #variables spécifiques à la salle
    espace_x_porte = 0.7 #espace entre la première porte et la première table
    espace_x = 0.85 #espace qui separe dans la direction x deux tables
    espace_y = 1.1 #espace qui separe dans la direction y deux tables
    largeur_bureau_Gs = 0.5
    longueur_bureau_Gs = 1.25
    espace_agent_table = 0.17 #distance entre le bord de la table et l'élève

    """Partie à modifier pour les différentes simulations """
    portes_ouvertes = [1,2]
    partie_gauche = [1,2,3,4]
    partie_droite = [1,2,3]
    """ """

    #simulation de la salle
    Gd_salle_2_portes = Salle_233_235(x0,y0,largeur_porte,portes_ouvertes) 
    Portes_Gs = Gd_salle_2_portes[1] #on récupère les portes
    Gd_salle_2_portes = Gd_salle_2_portes[0]

    Gd_salle_tables = tables_salle_233_235(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs)
    Obstacles_Gd_salle = Gd_salle_2_portes + Gd_salle_tables

    eleves_Gs = Salle_233_235_occupation(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,
                                    longueur_bureau_Gs,espace_agent_table,partie_gauche,partie_droite,sigma,epsilon)

    grande_salleTest = Environnement(x0+14,y0+6,Nx,Ny,dt,Obstacles_Gd_salle,eleves_Gs,Portes_Gs)
    #==============================================================================


    #salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)

    agents_positions = list(salleTest.maj_turns(nombreT))


    i = 0
    filename = "agents_positions.csv"
    while os.path.exists("data/" + filename):
        filename = "agents_positions_" + str(i) + ".csv"
        i += 1
    with open("data/" + filename, 'w') as csvfile:
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
    ani.save("media/" + filename + ".mp4")
    plt.show()
