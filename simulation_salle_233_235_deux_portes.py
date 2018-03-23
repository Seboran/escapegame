# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:07:26 2018

@author: marie_000
"""

from Fonctions_classes import *

def Salle_233_235_deux_portes(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs):
    murs_grd_salle = []
    murs_grd_salle.append(Obstacle([[x0,y0],[x0+0.1,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte,y0],[x0+0.1+largeur_porte+5.8,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte+4.2,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0],[x0,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0+5.8],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0+0.1,y0],[x0+0.1,y0+0.8]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte+5.8,y0],[x0+0.1+largeur_porte+5.8,y0+0.8]]))
    murs_grd_salle.append(Obstacle([[x0+5.9,y0+5.8],[x0+5.9,y0+5.6]]))

    tables_Gs = []
    #ajout des tables dans la parte gauche de la salle
    #simulation des tables collées au mur 
    for i in range(3):
        x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau_Gs + espace_x)
        x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau_Gs + i*(espace_x + largeur_bureau_Gs)
        y1 = y0
        y2 = y1 + longueur_bureau_Gs
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
    #simulation des tables au centre de la pièce
    for i in range(4):
        x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau_Gs + espace_x)
        x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau_Gs + i*(espace_x + largeur_bureau_Gs)
        y1 = y0 + longueur_bureau_Gs + espace_y
        y2 = y1 + longueur_bureau_Gs
        y3 = y2 + longueur_bureau_Gs
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]]),
                                 Obstacle([[x1,y2],[x1,y3]]), Obstacle([[x1,y3],[x2,y3]]),
                                 Obstacle([[x2,y3],[x2,y2]]), Obstacle([[x2,y2],[x1,y2]])]
    
    #ajout des tables dans la parte droite de la salle   
    #simulation des tables collées au mur
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2
        x2 = x1 - longueur_bureau_Gs
        y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
        y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
    
    #simulation des tables au centre de la pièce
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(2*longueur_bureau_Gs + espace_x)
        x2 = x1 - longueur_bureau_Gs
        y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
        y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(longueur_bureau_Gs + espace_x)
        x2 = x1 - longueur_bureau_Gs
        if(i == 2):
            x1 = x0+0.1+5.8+2*largeur_porte+4.2 - longueur_bureau_Gs
            x2 = x1 - longueur_bureau_Gs
        y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
        y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
    
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
        
        
    obstacles = murs_grd_salle + tables_Gs
    return obstacles