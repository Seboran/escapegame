# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:07:26 2018

@author: marie_000
"""

import numpy as np
from utils import *


#==============================================================================
#Fonctions de simulation de la pièce
#==============================================================================
def Salle_233_235_deux_portes(x0,y0,largeur_porte):
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
    
    #simulation des portes
    porte_1 = Porte([x0+0.1,y0-1], [x0+0.1+largeur_porte,y0-1])
    porte_2 = Porte([x0+0.1+largeur_porte+5.8,y0-1], [x0+0.1+5.8+2*largeur_porte,y0-1])
    return murs_grd_salle, [porte_1,porte_2]



def Salle_233_235_porte_gauche(x0,y0,largeur_porte):
    murs_grd_salle = []
    murs_grd_salle.append(Obstacle([[x0,y0],[x0+0.1,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte,y0],[x0+0.1+largeur_porte+10+largeur_porte,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte+4.2,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0],[x0,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0+5.8],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0+0.1,y0],[x0+0.1,y0+0.8]]))
    murs_grd_salle.append(Obstacle([[x0+5.9,y0+5.8],[x0+5.9,y0+5.6]]))
    
    #simulation des portes
    porte = Porte([x0+0.1,y0-1], [x0+0.1+largeur_porte,y0-1])
    return murs_grd_salle, [porte]



def Salle_233_235_porte_droite(x0,y0,largeur_porte):
    murs_grd_salle = []
    murs_grd_salle.append(Obstacle([[x0,y0],[x0+0.1+largeur_porte+5.8,y0]]))

    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte+4.2,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0],[x0,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0+5.8],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte+5.8,y0],[x0+0.1+largeur_porte+5.8,y0+0.8]]))
    murs_grd_salle.append(Obstacle([[x0+5.9,y0+5.8],[x0+5.9,y0+5.6]]))
    
    #simulation des portes
    porte = Porte([x0+0.1+largeur_porte+5.8,y0-1], [x0+0.1+5.8+2*largeur_porte,y0-1])
    return murs_grd_salle, [porte]



def tables_salle_233_235(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau,longueur_bureau):
    tables_Gs = []
    #ajout des tables dans la parte gauche de la salle
    #simulation des tables collées au mur 
    for i in range(3):
        x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau + espace_x)
        x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau + i*(espace_x + largeur_bureau)
        y1 = y0
        y2 = y1 + longueur_bureau
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
    #simulation des tables au centre de la pièce
    for i in range(4):
        x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau + espace_x)
        x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau + i*(espace_x + largeur_bureau)
        y1 = y0 + longueur_bureau + espace_y
        y2 = y1 + longueur_bureau
        y3 = y2 + longueur_bureau
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]]),
                                 Obstacle([[x1,y2],[x1,y3]]), Obstacle([[x1,y3],[x2,y3]]),
                                 Obstacle([[x2,y3],[x2,y2]]), Obstacle([[x2,y2],[x1,y2]])]    
    #ajout des tables dans la parte droite de la salle   
    #simulation des tables collées au mur
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2
        x2 = x1 - longueur_bureau
        y1 = y0 + espace_y + i*(largeur_bureau + espace_y)
        y2 = y0 + espace_y + largeur_bureau +i*(largeur_bureau + espace_y)
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]    
    #simulation des tables au centre de la pièce
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(2*longueur_bureau + espace_x)
        x2 = x1 - longueur_bureau
        y1 = y0 + espace_y + i*(largeur_bureau + espace_y)
        y2 = y0 + espace_y + largeur_bureau +i*(largeur_bureau + espace_y)
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
    for i in range(3):
        x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(longueur_bureau + espace_x)
        x2 = x1 - longueur_bureau
        if(i == 2):
            x1 = x0+0.1+5.8+2*largeur_porte+4.2 - longueur_bureau
            x2 = x1 - longueur_bureau
        y1 = y0 + espace_y + i*(largeur_bureau + espace_y)
        y2 = y0 + espace_y + largeur_bureau +i*(largeur_bureau + espace_y)
    
        tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
                                 Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]         
    return tables_Gs

#==============================================================================
#Fonction de simulation des élèves dans la pièce
#==============================================================================
def Salle_233_235_occupation(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau,
                                  longueur_bureau,espace_agent_table,rangs_gauche,rangs_droite,sigma,epsilon):
    #les variables rangs_gauche et rangs_droite permettent d'indiquer sur quels rangs de la partie gauche et droite
    #de la salle on souhaite placer des élèves
    eleves = []
    for j in [0.25,0.75]:
        #ajout des élèves dans la partie gauche de la salle (on place un agent à chaque place)
        for i in rangs_gauche:
            if(i !=4):
                x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau + (i-1)*(largeur_bureau + espace_x)
                eleve = Agent(np.array([x+espace_agent_table,y0+j*longueur_bureau]), 1., sigma, epsilon*2, 'eleve')
                eleves.append(eleve)
        for i in rangs_gauche:
             x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau + (i-1)*(largeur_bureau + espace_x)
             y = y0+longueur_bureau+espace_y+j*longueur_bureau 
             eleve = Agent(np.array([x+espace_agent_table,y]), 1., sigma, epsilon*2, 'eleve')
             eleves.append(eleve)
        for i in rangs_gauche:
             x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau + (i-1)*(largeur_bureau + espace_x)
             y = y0+longueur_bureau+espace_y+j*longueur_bureau + longueur_bureau 
             eleve = Agent(np.array([x+espace_agent_table,y]), 1., sigma, epsilon*2, 'eleve')
             eleves.append(eleve)
        #ajout des élèves dans la partie droite de la salle (on place un agent à chaque place)
        for i in rangs_droite:
            x =  x0+0.1+5.8+2*largeur_porte+4.2 - j*longueur_bureau
            y =  y0 + espace_y + (i-1)*(largeur_bureau + espace_y) + largeur_bureau
            eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
            eleves.append(eleve)
        for i in rangs_droite:
             x = x0+0.1+5.8+2*largeur_porte+4.2 - (espace_x+2*longueur_bureau) - j*longueur_bureau
             y =  y0 + espace_y + (i-1)*(largeur_bureau + espace_y) + largeur_bureau
             eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
             eleves.append(eleve)
        for i in rangs_droite:
            x = x0+0.1+5.8+2*largeur_porte+4.2 - (espace_x+longueur_bureau) - j*longueur_bureau
            if((i-1) == 2):
                x = x0+0.1+5.8+2*largeur_porte+4.2 - longueur_bureau - j*longueur_bureau
            y =  y0 + espace_y + (i-1)*(largeur_bureau + espace_y) + largeur_bureau
            eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
            eleves.append(eleve)
            
        return eleves
    

