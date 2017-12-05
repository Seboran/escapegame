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



class Couple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
class Agent:
    def __init__(self, positionBase, vitesseBase, sigma, epsilon):
        self.vitesseBase = vitesseBase
        self.sigma = sigma
        self.epsilon = epsilon
        self.vitesse = [0, 0]
        self.position = positionBase
        
class Porte:
    def __init__(self, positionGauche, positionDroite):
        self.positionGauche = positionGauche
        self.positionDroite = positionDroite
        self.positionCentre=Couple(0.5*(self.positionDroite.x+self.positionGauche.x),
                                   0.5*(self.positionDroite.y+self.positionGauche.y))
        
class Obstacle:
    def __init__(self, sommets):
        # Sommets est un couple de couples
        self.sommets = sommets
        
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
            nx = meter_to_int(agent.position.x, Nx, Lx)
            ny = meter_to_int(agent.position.y, Ny, Ly)
            
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
                agent.position.x += self.dt * agent.vitesse[0]
                agent.position.y += self.dt * agent.vitesse[1]
                
        maj_vitesse_agents()
        maj_position_agents()
        
        
    def afficher(self, figure, axe):
        x = []
        y = []
        axe.set_xlim(0, self.Lx)
        axe.set_ylim(0, self.Ly)
        for agent in self.agents:
            x.append(agent.position.x)
            y.append(agent.position.y)
        
        axe.scatter(x, y)
        for obstacle in self.obstacles:

            x = []
            y = []
            for sommet in obstacle.sommets:
                x.append(sommet.x)
                y.append(sommet.y)
                
            axe.plot(x, y, color = '#000000')
            
        for porte in self.portes:
            pos_porte = porte.positionCentre
            plt.plot(pos_porte.x, pos_porte.y, 'x')
    
def fintention(agent, portes):
    
    #On cherche la porte la plus proche
    
    vect=[portes[0].positionCentre.x-agent.position.x,portes[0].positionCentre.y-agent.position.y]
    vect=vect/np.linalg.norm(vect)
    
    for porte in portes[1:-1]:
        
        vect_test=[porte.positionCentre.x-agent.position.x,porte.positionCentre.y-agent.position.y]
        vect_test=vect_test/np.linalg.norm(vect_test)
        
        if np.linalg.norm(vect_test) < np.linalg.norm(vect):
            
            vect = vect_test
        
    return vect
    

# Premier exemple

Lx = 10.
Ly = 15.
Nx = 400
Ny = 400
dt = 1.

marie = Agent(Couple(5., 5.), 2., 1., 1.)
nirina = Agent(Couple(7., 2.), 2., 2., 2.)

            


      



# Murs d'exemple

mur0 = Obstacle([Couple(4., 1.), Couple(1., 1.)])
mur1 = Obstacle([Couple(1., 1.), Couple(1., 14.)])
mur2 = Obstacle([Couple(1., 14.), Couple(9., 14.)])
mur3 = Obstacle([Couple(9., 14.), Couple(9., 1.)])
mur4 = Obstacle([Couple(9., 1.), Couple(6., 1.)])



obstacles = [mur0, mur1, mur2, mur3, mur4]

porte = Porte(Couple(4., 1.), Couple(6., 1.))

agents = [marie, nirina]
portes = [porte]

TEST=fintention(marie, portes)

salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)


fig, ax = plt.subplots(1,1)

plt.show()

salleTest.afficher(fig, ax) 

for i in range(10):
    salleTest.maj()
    salleTest.afficher(fig, ax)
    time.sleep(1)














