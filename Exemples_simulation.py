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


#==============================================================================
#SIMULATIONS AMPHI
#==============================================================================
largeur_porte = 0.80 #on met la largeur de la demi-porte
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
#================================================================================
#simulation 2 : 4 demi-portes ouvertes - moitié amphi rempli (3 premières lignes)
#=================================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2] 
occupation_colonne = range(16) 
#================================================================================
#simulation 3 : 4 demi-portes ouvertes - moitié amphi rempli (3 dernières lignes)
#=================================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [3,4,5] 
occupation_colonne = range(16) 
#================================================================================
#simulation 4 : 4 demi-portes ouvertes - moitié amphi rempli partie droite
#=================================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2,3,4,5]
occupation_colonne = range(8) 
#================================================================================
#simulation 5 : 4 demi-portes ouvertes - moitié amphi rempli partie centrale
#=================================================================================
portes_ouvertes = [1,2,3,4] 
occupation_ligne = [0,1,2,3,4,5] 
occupation_colonne = [4,5,6,7,8,9,10,11]



#============================================================
#Simulation amphi - partie invariable
#============================================================
murs_amphi = Amphi(x0,y0,largeur_porte,portes_ouvertes)
portes_amphi = murs_amphi[1]
murs_amphi = murs_amphi[0]
tables_amphi = table_amphi(x0,y0,espace_x,espace_y,nbr_rangees,largeur_table,longueur_table,espace_table)
eleves_amphi = amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne,occupation_ligne,sigma,epsilon)