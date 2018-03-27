# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:07:26 2018

@author: marie_000
"""

import numpy as np
from enviro_escape import *


#==============================================================================
#Fonctions de simulation de la pièce 233/235
#==============================================================================
def Salle_233_235(x0,y0,largeur_porte,ouvertures):
    ouvertures.sort()
    Portes =  []
    murs_grd_salle = []
    murs_grd_salle.append(Obstacle([[x0,y0],[x0+0.1,y0]]))
    if(len(ouvertures)>=1 and ouvertures[0] == 1):
        #la première porte est ouverte
        murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte,y0],[x0+0.1+largeur_porte+5.8,y0]]))
        murs_grd_salle.append(Obstacle([[x0+0.1,y0],[x0+0.1,y0+0.8]]))
        porte_1 = Porte([x0+0.1,y0-1], [x0+0.1+largeur_porte,y0-1])
        Portes.append(porte_1)
        del ouvertures[0]
    else :
        #la première porte est fermée
        murs_grd_salle.append(Obstacle([[x0+0.1,y0],[x0+0.1+largeur_porte+5.8,y0]]))
    if(len(ouvertures)>=1 and ouvertures[0] == 2):
        #la deuxième porte est ouverte
        murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0]]))
        murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte+5.8,y0],[x0+0.1+largeur_porte+5.8,y0+0.8]]))
        porte_2 = Porte([x0+0.1+largeur_porte+5.8,y0-1], [x0+0.1+5.8+2*largeur_porte,y0-1])
        Portes.append(porte_2)
        del ouvertures[0]
    else:
        #la deuxième porte est fermée
        murs_grd_salle.append(Obstacle([[x0+0.1+5.8,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0]]))
        
    murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte+4.2,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0],[x0,y0+5.8]]))
    murs_grd_salle.append(Obstacle([[x0,y0+5.8],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))   
    murs_grd_salle.append(Obstacle([[x0+5.9,y0+5.8],[x0+5.9,y0+5.6]]))
    
    return murs_grd_salle, Portes


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
#Fonction de simulation des élèves dans la pièce 233/235
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





#==============================================================================
#Fonctions de simulation de l'amphi
#==============================================================================
def Amphi(x0,y0,largeur_porte,ouvertures): 
    ouvertures.sort()
    murs_amphi = []
    Portes =  []
    murs_amphi.append(Obstacle([[x0,y0],[x0+0.4, y0]]))
    if(len(ouvertures) >= 1 and ouvertures[0] == 1):
        #la première demi-porte est ouverte
        if(len(ouvertures) >=2 and ouvertures[1] == 2):
            #la deuxième demi-porte est ouverte
            murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6,y0]]))
            porte_1 = Porte([x0+0.4,y0-1], [x0+0.4+2*largeur_porte,y0-1])
            Portes.append(porte_1)
            del ouvertures[0]
            del ouvertures[0]
        else:#seule la première demi_porte est ouverte
            murs_amphi.append(Obstacle([[x0+0.4+largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6,y0]]))
            porte_1 = Porte([x0+0.4,y0-1], [x0+0.4+largeur_porte,y0-1])
            Portes.append(porte_1)
            del ouvertures[0]
    elif(len(ouvertures) >=1 and ouvertures[0] == 2):
        #la première demi-porte est fermée mais la deuxième est ouverte
        murs_amphi.append(Obstacle([[x0+0.4,y0],[x0+0.4+largeur_porte,y0]]))
        murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6,y0]]))
        porte_1 = Porte([x0+0.4+largeur_porte,y0-1], [x0+0.4+2*largeur_porte,y0-1])
        Portes.append(porte_1)
        del ouvertures[0]
    else:
        #les deux premières demi-portes sont fermées
        murs_amphi.append(Obstacle([[x0+0.4,y0],[x0+0.4+2*largeur_porte+9.6,y0]]))
    if(len(ouvertures) >=1 and ouvertures[0] == 3):
        #la troisième demi-porte est ouverte
        if(len(ouvertures) >= 2 and ouvertures[1] == 4):
            #la quatrième demi-porte est ouverte
            murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+2*largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0]]))
            porte_2 = Porte([x0+0.4+2*largeur_porte+9.6,y0-1], [x0+0.4+9.6+4*largeur_porte,y0-1])
            Portes.append(porte_2)
            del ouvertures[0]
            del ouvertures[0]
        else: #seule la troisième demi-porte est ouverte
            murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0]]))
            porte_2 = Porte([x0+0.4+2*largeur_porte+9.6,y0-1], [x0+0.4+9.6+3*largeur_porte,y0-1])
            Portes.append(porte_2)
            del ouvertures[0]
    elif(len(ouvertures) >=1 and ouvertures[0] == 4):
        #la troisième demi-porte est fermée mais la quatrième est ouverte
        murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6,y0],[x0+0.4+2*largeur_porte+9.6+largeur_porte,y0]]))
        murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+2*largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0]]))
        porte_2 = Porte([x0+0.4+3*largeur_porte+9.6,y0-1], [x0+0.4+9.6+4*largeur_porte,y0-1])
        Portes.append(porte_2)
        del ouvertures[0]
    else:
        #les deux dernières demi-portes sont fermées
        murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0]]))            
            
    murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0+8.1]]))
    murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+2*largeur_porte+0.2,y0+8.1],[x0,y0+8.1]]))
    murs_amphi.append(Obstacle([[x0,y0+8.1],[x0,y0]]))
    murs_amphi.append(Obstacle([[x0+0.4,y0],[x0+0.4,y0+0.8]]))
    murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte,y0],[x0+0.4+2*largeur_porte,y0+0.8]]))
    murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6,y0],[x0+0.4+2*largeur_porte+9.6,y0+0.8]]))
    murs_amphi.append(Obstacle([[x0+0.4+2*largeur_porte+9.6+2*largeur_porte,y0],[x0+0.4+2*largeur_porte+9.6+2*largeur_porte,y0+0.8]]))
    
    return murs_amphi,Portes



def table_amphi(x0,y0,espace_x,espace_y,nbr_rangees,largeur_table,longueur_table,espace_table):
    tables = []
    for i in range(nbr_rangees):
        X0 = x0+espace_x
        Y0 = y0+espace_y
        tables.append(Obstacle([[X0,Y0+i*(largeur_table+espace_table)],[X0+longueur_table,Y0+i*(largeur_table+espace_table)]]))
        tables.append(Obstacle([[X0,Y0+i*(largeur_table+espace_table)],[X0,Y0+largeur_table+i*(largeur_table+espace_table)]]))
        tables.append(Obstacle([[X0,Y0+largeur_table+i*(largeur_table+espace_table)],[X0+longueur_table,Y0+largeur_table+i*(largeur_table+espace_table)]]))
        tables.append(Obstacle([[X0+longueur_table,Y0+i*(largeur_table+espace_table)],[X0+longueur_table,Y0+largeur_table+i*(largeur_table+espace_table)]]))
    
    #rajout du bureau du prof
    tables.append(Obstacle([[x0+espace_x+0.25*longueur_table,y0+espace_y/2.0],[x0+espace_x+0.75*longueur_table,y0+espace_y/2.0]]))
    tables.append(Obstacle([[x0+espace_x+0.25*longueur_table,y0+espace_y/2.0+largeur_table],[x0+espace_x+0.75*longueur_table,y0+espace_y/2.0+largeur_table]]))
    tables.append(Obstacle([[x0+espace_x+0.25*longueur_table,y0+espace_y/2.0],[x0+espace_x+0.25*longueur_table,y0+espace_y/2.0+largeur_table]]))
    tables.append(Obstacle([[x0+espace_x+0.75*longueur_table,y0+espace_y/2.0],[x0+espace_x+0.75*longueur_table,y0+espace_y/2.0+largeur_table]]))

    return tables

#==============================================================================
#Fonction de simulation des élèves dans l'amphi
#==============================================================================
def amphi_occupation(x0,y0,espace_x,espace_y,longueur_table,largeur_table,espace_table_eleve,espace_table,occupation_colonne,occupation_ligne,
                     sigma,epsilon):
    
    eleves = []
    X0 = x0+espace_x
    Y0 = y0+espace_y
    position = (longueur_table/16)/2
    for i in occupation_colonne:
        print("colonne" + repr(i))
        for j in occupation_ligne:
            print("ligne" + repr(j))
            eleve = Agent(np.array([X0+position+2*i*position,Y0+largeur_table+espace_table_eleve+j*(espace_table+largeur_table)]), 1., sigma, epsilon*2, 'eleve')                
            eleves.append(eleve)
    return eleves

