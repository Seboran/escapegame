# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:29:35 2017

@author: nirin
"""

# test

# Je rajoute quelques lignes de commentaires

import numpy as np
import matplotlib.pylab as plt
import time
import scipy

plt.close("all")

numero_agent = 1
numero_porte = -1

        
        
class Agent:
    def __init__(self, positionBase, vitesseBase, sigma, epsilon):
        self.vitesseBase = vitesseBase
        self.sigma = sigma
        self.epsilon = epsilon
        self.vitesse = np.array([0, 0])
        self.position = np.array(positionBase)
        
class Porte:
    def __init__(self, positionGauche, positionDroite):
        self.positionGauche = np.array(positionGauche)
        self.positionDroite = np.array(positionDroite)

        self.positionCentre= 0.5*(self.positionDroite+self.positionGauche)

        
class Obstacle:
    def __init__(self, sommets):
        # Sommets est un couple de couples
        self.sommets = np.array(sommets)
    def __del__(self):
        del self.sommets

        
class Grille:
    def __init__(self, Nx, Ny): 
        self.tab = [[0]*Nx for _ in range(Ny)]
    
        
class Environnement:
    def __init__(self, Lx, Ly, Nx, Ny, dt, obstacles, agents, portes):
        self.obstacles = obstacles
        self.agents = agents
        self.portes = portes
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dt = dt
        self.grille = Grille(Nx, Ny)
        
        def meter_to_int(x, N, L):
            return int(x * N / L) # dégueulasse
        
        for obstacle in obstacles:
            # TODO
            2;
            
        for agent in agents:
            nx = meter_to_int(agent.position[0], Nx, Lx)
            ny = meter_to_int(agent.position[1], Ny, Ly)
            
            self.grille.tab[nx][ny] = numero_agent
            
        for porte in portes:
            #Todo
            
            2;
            
    
    
    
    
            
    def maj(self):
        def maj_vitesse_agents():
        
            def maj_vitesse_agent_intention(agent):
                force = fintention(agent, self.portes)
                agent.vitesse += force
            
            for agent in self.agents:
                agent.vitesse = [0, 0]
                maj_vitesse_agent_intention(agent)
            
        def maj_position_agents():
            for agent in agents:
                agent.position = agent.position + self.dt * agent.vitesse
                
        maj_vitesse_agents()
        maj_position_agents()
        
        
    def afficher(self, figure, axe):
        x = []
        y = []
        axe.set_xlim(0, self.Lx)
        axe.set_ylim(0, self.Ly)
        for agent in self.agents:
            x.append(agent.position[0])
            y.append(agent.position[1])
        
        axe.scatter(x, y)
        for obstacle in self.obstacles:

            x = []
            y = []
            for sommet in obstacle.sommets:
                x.append(sommet[0])
                y.append(sommet[1])
                
            axe.plot(x, y, color = '#000000')
            
        for porte in self.portes:
            pos_porte = porte.positionCentre
            plt.plot(pos_porte[0], pos_porte[1], 'x')

def build_walls(Lx,Ly,portes):
#On suppose que les portes sont bien définies
    
    liste_murs=[]
    murd=[0,Ly]
    murg=[0,Ly]
    murh=[0,Lx]
    murb=[0,Lx]
    
    for porte in portes:
        
        #Si x est le même : porte à gauche ou à droite
        if porte.positionGauche[0]==porte.positionDroite[0]:
            
            xp=porte.positionGauche[0]
            
            #La porte est à gauche
            if xp==0:
                murg+=[porte.positionGauche[1],porte.positionDroite[1]]
            #La porte est à droite
            else:
                murd+=[porte.positionGauche[1],porte.positionDroite[1]]
            
        #Sinon y est le même : porte en haut ou en bas
        else:
            
            yp=porte.positionGauche[1]
            
            #La porte est en bas
            if yp==0:
                murb+=[porte.positionGauche[0],porte.positionDroite[0]]
            #La porte est en haut
            else:
                murh+=[porte.positionGauche[0],porte.positionDroite[0]]
    
    murb=sorted(murb)
    murh=sorted(murh)
    murd=sorted(murd)
    murg=sorted(murg)
    
    
    #Build left walls
    liste_murs+=[Obstacle([[0,murg[0]],[0,murg[1]]])]
    
    for i in range(2,len(murg),2):
        
        liste_murs.append(Obstacle([[0,murg[i]],[0,murg[i+1]]]))
       
    #Build right walls
    liste_murs.append(Obstacle([[0,murd[0]],[0,murd[1]]]))
    
    for i in range(2,len(murd),2):
        
        liste_murs.append(Obstacle([[Lx,murd[i]],[Lx,murd[i+1]]]))
        
    #Build bottom walls
    liste_murs.append(Obstacle([[murb[0],0],[murb[1],0]]))
    
    for i in range(2,len(murb),2):
        
        liste_murs.append(Obstacle([[murb[i],0],[murb[i+1],0]]))
    
    #Build top walls
    liste_murs.append(Obstacle([[murh[0],Ly],[murh[1],Ly]]))
    
    for i in range(2,len(murh),2):
        
        liste_murs.append(Obstacle([[murh[i],0],[murh[i+1],0]]))
            
    return liste_murs


def fintention(agent, portes):

# Intention naturellle pour un agent d'aller vers la porte la plus proche

    
    vect=portes[0].positionCentre-agent.position
    vect=vect/np.linalg.norm(vect)
    
    for porte in portes[1:-1]:
        
        vect_test=portes.positionCentre-agent.position
        vect_test=vect_test/np.linalg.norm(vect_test)
        
        if np.linalg.norm(vect_test) < np.linalg.norm(vect):
            
            vect = vect_test
        
    return agent.vitesseBase * np.array(vect)
    
def Dpotentiel(r,sigma,epsilon):
#La dérivée du potentiel de répulsion
    
    if r<2**(1/6)*sigma:
        
        return 4*epsilon*(-12*(sigma/r)**12/r+6*(sigma/r)**6/r)
    
    else:
        return 0
    
def fagent(agent1,agent2):
#Force de répulsion entre deux agents

    sigma=agent1.sigma
    epsilon=agent1.epsilon
    r=np.linalg.norme(agent1.position-agent2.position)
    
    return -Dpotentiel(r,sigma,epsilon)
    
    


# Premier exemple

Lx = 10.
Ly = 15.
Nx = 400
Ny = 400
dt = 0.5

marie = Agent([5,5], 2., 1., 1.)
nirina = Agent([7,2], 2., 2., 2.)


# Murs d'exemple

#==============================================================================
# mur0 = Obstacle([[4,1],[1,1]])
# mur1 = Obstacle([[1,1], [1,14]])
# mur2 = Obstacle([[1,14], [9,14]])
# mur3 = Obstacle([[9,14], [9,1]])
# mur4 = Obstacle([[9,1], [6,1]])
# 
# 
# 
# obstacles = [mur0, mur1, mur2, mur3, mur4]
#==============================================================================

porte = Porte([4,0], [6,0])

agents = [marie, nirina]
portes = [porte]
obstacles=build_walls(Lx,Ly,portes)

salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)


fig, ax = plt.subplots(1,1)

plt.show()

salleTest.afficher(fig, ax) 

for i in range(10):
    salleTest.maj()
    salleTest.afficher(fig, ax)
    















