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
import random

plt.close("all")



numero_agent = 1
numero_porte = -1
seuil_porte = 0.1
        
class Agent:
    def __init__(self, positionBase, vitesseBase, sigma, epsilon, nom = '', color = ''):
        self.vitesseBase = vitesseBase
        self.sigma = sigma
        self.epsilon = epsilon
        self.vitesse = np.array([0, 0])
        self.position = positionBase
        r = lambda: random.randint(0,255)
        if color == '':
            self.color = ('#%02X%02X%02X' % (r(),r(),r()))
        else:
            self.color = color
        self.alive = True
        self.nom = nom
        # note : si quelqu'un traverse une porte il peut toujours gêner la sortie d'une salle
        
        
    def distance(self, element):
        if type(element) == Agent:
            return np.linalg.norm(self.position - element.position)
        if type(element) == Porte:
            return np.linalg.norm(self.position - porte.positionCentre)
    
    
        
class Porte:
    def __init__(self, positionGauche, positionDroite):
        self.positionGauche = positionGauche
        self.positionDroite = positionDroite

        self.positionCentre= 0.5*(self.positionDroite+self.positionGauche)

        
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
            nx = meter_to_int(agent.position[0], Nx, Lx)
            ny = meter_to_int(agent.position[1], Ny, Ly)
            
            self.grille.tab[nx][ny] = numero_agent
            
        for porte in portes:
            #Todo
            
            2;
            
    
    
    
    
            
    def maj(self):
        #========================================================
        def maj_vitesse_agents():
            #========================================================
            def maj_vitesse_agent_intention(agent):
                force = fintention(agent, self.portes)
                agent.vitesse += force * self.dt
            #========================================================
            #========================================================
            def maj_vitesse_agent_repulsion(agent):
                for agent_i in self.agents:
                    
                    if agent_i != agent:
                        
                        force = fagent(agent, agent_i)
                        agent.vitesse += force * self.dt
            #========================================================
            for agent in self.agents:
                if agent.alive:
                    agent.vitesse = np.array([0., 0.])
                    
                    maj_vitesse_agent_intention(agent)
                    maj_vitesse_agent_repulsion(agent)
        #========================================================
        
        def maj_position_agents():
            for agent in self.agents:
                if agent.alive:
                    agent.position = agent.position + self.dt * agent.vitesse
                
        def maj_agents_alive():
            for agent in self.agents:
                for porte in self.portes:
                    if agent.distance(porte) < seuil_porte:
                        agent.alive = False

        maj_vitesse_agents()
        maj_position_agents()
        maj_agents_alive()
    
        
    def afficher(self, figure, axe):
        x = []
        y = []
        color = []
        axe.set_xlim(0, self.Lx)
        axe.set_ylim(0, self.Ly)
        for agent in self.agents:
            x.append(agent.position[0])
            y.append(agent.position[1])
            color.append(agent.color)
            
        
        axe.scatter(x, y, c = color)
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
    r = agent1.distance(agent2)
    
    vecteur_unitaire = (agent1.position - agent2.position) / r
    
    if agent2.alive:
        amplitude = -Dpotentiel(r,sigma,epsilon)
    else:
        amplitude = 0.
    
    return amplitude * vecteur_unitaire
    
    


# Premier exemple

Lx = 10.
Ly = 15.
Nx = 400
Ny = 400
dt = 0.2
sigma = 1.
epsilon = 1.0

dts = np.linspace(0.1, 1, 10)
for dt in dts:
    
    
    marie = Agent(np.array([5,5]), 1., sigma, epsilon, 'marie')
    nirina = Agent(np.array([np.sqrt(2.)/2. * 5., np.sqrt(2.)/2. * 5.]), 1., sigma, epsilon, 'nirina')
    luc = Agent(np.array([10 - np.sqrt(2.)/2. * 5., np.sqrt(2.)/2. * 5.]), 1., sigma, epsilon, 'luc')
    
    # Murs d'exemple
    
    mur0 = Obstacle([np.array([4,1]),np.array([1,1])])
    mur1 = Obstacle([np.array([1,1]), np.array([1,14])])
    mur2 = Obstacle([np.array([1,14]), np.array([9,14])])
    mur3 = Obstacle([np.array([9,14]), np.array([9,1])])
    mur4 = Obstacle([np.array([9,1]), np.array([6,1])])
    
    
    
    obstacles = [mur0, mur1, mur2, mur3, mur4]
    
    porte = Porte(np.array([4,1]), np.array([6,1]))
    
    agents = [marie, nirina, luc]
    portes = [porte]
    
    TEST=fintention(marie, portes)
    
    salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)
    
    
    fig, ax = plt.subplots(1,1)
    
    plt.show()
    
    salleTest.afficher(fig, ax) 
    
    for i in range(int(10 / dt)):
        salleTest.maj()
    
        salleTest.afficher(fig, ax)
        
    
    
    
    
    
    
    
    
    
    
    




