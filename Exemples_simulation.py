# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:50:19 2018

@author: marie_000
"""

"""Voici quelques exemples de simulations que l'on peut faire (à copier dans le fichier main") """


#variables générales utiles à la simulation
#position de la pièce de l'angle en bas à gauche dans la fenêtre d'affichage
x0 = 1.0 #décalage en x
y0 = 3.0 #dacalage en y
sigma = 0.1
epsilon = 1.0
largeur_porte = 0.80  #largeur d'une petite porte ou d'une demi-porte d'amphi

#==============================================================================
#SIMULATIONS AMPHI
#==============================================================================
espace_x = 1.9 #esapce entre le mur de l'amphi et le bord gauche des tables 
espace_y = 2.3 #esapce entre le mur de l'amphi et le bord inférieur des tables
largeur_table = 0.3 #largueur des tables dans l'amphi
longueur_table = 9.95 #longueur des tables de l'amphi
espace_table = 0.6 #esapce entre deux tables dans l'amphi (on considère que les tables sont equidistantes)
nbr_rangees = 6 #nombre de rangées de tables dans l'amphi
espace_table_eleve = 0.19 #espace entre le bord de la table et l'élève


#==========================================================
#simulation 1 : 4 demi-portes ouvertes - occupation totale
#==========================================================

portes_ouvertes = [1,2,3,4]
occupation_ligne = range(nbr_rangees) 
occupation_colonne = range(16) 
#==============================================================================
#simulation 2 : 4 demi-portes ouvertes - moitié amphi rempli (3 premières lignes)
#==============================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2] 
occupation_colonne = range(16) 
#==============================================================================
#simulation 3 : 4 demi-portes ouvertes - moitié amphi rempli (3 dernières lignes)
#==============================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [3,4,5] 
occupation_colonne = range(16) 
#==============================================================================
#simulation 4 : 4 demi-portes ouvertes - moitié amphi rempli partie droite
#==============================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = range(8) 
#==============================================================================
#simulation 5 : 4 demi-portes ouvertes - moitié amphi rempli partie centrale
#==============================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2,3,4,5] 
occupation_colonne = [4,5,6,7,8,9,10,11]
#==============================================================================
#simulation 6 : 2 demi-portes ouvertes (une de chaque côté) - amphi plein
#==============================================================================
portes_ouvertes = [1,3] 
occupation_ligne = [0,1,2,3,4,5] 
occupation_colonne = range(16)
#==============================================================================
#simulation 7 : 2 demi-portes ouvertes (une de chaque côté) - moitié amphi rempli (3 premières lignes)
#==============================================================================
portes_ouvertes = [1,3]
occupation_ligne = [0,1,2]
occupation_colonne = range(16)
#==============================================================================
#simulation 8 : 2 demi-portes ouvertes (une de chaque côté) - moitié amphi rempli (3 dernières lignes)
#==============================================================================
portes_ouvertes = [1,3]
occupation_ligne = [3,4,5]
occupation_colonne = range(16)
#==============================================================================
#simulation 9 : 2 demi-portes ouvertes (une de chaque côté) - moitié amphi rempli partie droite
#==============================================================================
portes_ouvertes = [1,3]
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = range(8) 
#==============================================================================
#simulation 10 : 2 demi-portes ouvertes (une de chaque côté) - moitié amphi rempli partie centrale
#==============================================================================
portes_ouvertes = [1,3]
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = [4,5,6,7,8,9,10,11]
#==============================================================================
#simulation 11 : 1 grosse porte ouverte - amphi plein
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [0,1,2,3,4,5] 
occupation_colonne = range(16)
#==============================================================================
#simulation 12 : 1 grosse porte ouverte - moitié amphi rempli (3 premières lignes)
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [0,1,2] 
occupation_colonne = range(16)
#==============================================================================
#simulation 13 : 1 grosse porte ouverte - moitié amphi rempli (3 dernières lignes)
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [3,4,5] 
occupation_colonne = range(16)
#==============================================================================
#simulation 14 : 1 grosse porte ouverte - moitié amphi rempli partie droite
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = range(8)
#==============================================================================
#simulation 15 : 1 grosse porte ouverte - moitié amphi rempli partie gauche
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = range(8,16) 
#==============================================================================
#simulation 16 : 1 grosse porte ouverte - moitié amphi rempli partie centrale
#==============================================================================
portes_ouvertes = [1,2]
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = [4,5,6,7,8,9,10,11]
#==============================================================================
#simulation 17 : 1 grosse porte ouverte - une personne sur deux (identique d'une ligne à l'autre)
#==============================================================================
portes_ouvertes = [1,2] #à modifier si on veut
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = [1,3,5,7,9,11,13,15]
#==============================================================================
#simulation 18 : 1 grosse porte ouverte - une personne sur deux (alternance entre ligne)
#==============================================================================
portes_ouvertes = [1,2] #à modifier si on veut
occupation_ligne_1 = [0,2,4]
occupation_colonne_1 = [1,3,5,7,9,11,13,15]
occupation_ligne_2 = [1,3,5]
occupation_colonne_2 = [2,4,6,8,10,12,14]
#modification de la simulation des élèves
eleves_amphi_1 = amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne_1,occupation_ligne_1,sigma,epsilon)
eleves_amphi_2 =  amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne_2,occupation_ligne_2,sigma,epsilon)
eleves_amphi = eleves_amphi_1 + eleves_amphi_2


#============================================================
#Simulation amphi - partie invariable
#============================================================
murs_amphi = Amphi(x0,y0,largeur_porte,portes_ouvertes)
portes_amphi = murs_amphi[1]
murs_amphi = murs_amphi[0]
tables_amphi = table_amphi(x0,y0,espace_x,espace_y,nbr_rangees,largeur_table,longueur_table,espace_table)
#Peut être modifier dans certains cas
eleves_amphi = amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne,occupation_ligne,sigma,epsilon)
#============================================================





#==============================================================================
#SIMULATIONS SALLE 235
#==============================================================================
espace_x_porte = 0.7 #espace entre la première porte et la première table
espace_x = 0.85 #espace qui separe dans la direction x deux tables
espace_y = 1.1 #espace qui separe dans la direction y deux tables
largeur_bureau_Gs = 0.5
longueur_bureau_Gs = 1.25
espace_agent_table = 0.17 #distance entre le bord de la table et l'élève

#==========================================================
#simulation 1 : 2 portes ouvertes - occupation totale
#==========================================================
portes_ouvertes = [1,2]
partie_gauche = [1,2,3,4]
partie_droite = [1,2,3]
#==========================================================
#simulation 2 : 2 portes ouvertes - occupation partie droite
#==========================================================
portes_ouvertes = [1,2]
partie_gauche = []
partie_droite = [1,2,3]
#==========================================================
#simulation 3 : 2 portes ouvertes - occupation partie gauche
#==========================================================
portes_ouvertes = [1,2]
partie_gauche = [1,2,3,4]
partie_droite = []
#==========================================================
#simulation 4 : porte gauche fermées - occupation totale
#==========================================================
portes_ouvertes = [2]
partie_gauche = [1,2,3,4]
partie_droite = [1,2,3]
#==========================================================
#simulation 5 : porte gauche fermées - occupation partie gauche
#==========================================================
portes_ouvertes = [2]
partie_gauche = [1,2,3,4]
partie_droite = []
#==========================================================
#simulation 6 : porte gauche fermées - occupation partie droite
#==========================================================
portes_ouvertes = [2]
partie_gauche = []
partie_droite = [1,2,3]
#==========================================================
#simulation 7 : porte droite fermées - occupation totale
#==========================================================
portes_ouvertes = [1]
partie_gauche = [1,2,3,4]
partie_droite = [1,2,3]
#==========================================================
#simulation 8 : porte droite fermées - occupation partie gauche
#==========================================================
portes_ouvertes = [1]
partie_gauche = [1,2,3,4]
partie_droite = []
#==========================================================
#simulation 9 : porte droite fermées - occupation partie droite
#==========================================================
portes_ouvertes = [1]
partie_gauche = []
partie_droite = [1,2,3]

#============================================================
#Simulation salle 235 - partie invariable
#============================================================
Gd_salle_2_portes = Salle_233_235(x0,y0,largeur_porte,portes_ouvertes) 
Portes_Gs = Gd_salle_2_portes[1]
Gd_salle_2_portes = Gd_salle_2_portes[0]
Gd_salle_tables = tables_salle_233_235(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs)
Obstacles_Gd_salle = Gd_salle_2_portes + Gd_salle_tables
eleves_Gs = Salle_233_235_occupation(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs,espace_agent_table,partie_gauche,partie_droite,sigma,epsilon)