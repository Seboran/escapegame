# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:29:35 2017

@author: nirin
"""

# test

# Je rajoute quelques lignes de commentaires

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
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
        #========================================================
        def maj_vitesse_agents():
            #========================================================
            def maj_vitesse_agent_intention(agent):
                force = fintention(agent, self.portes)
                agent.vitesse += force
            #========================================================
            #========================================================
            def maj_vitesse_agent_repulsion(agent):
                for agent_i in self.agents:
                    
                    if agent_i != agent:
                        
                        force = fagent(agent, agent_i)
                        agent.vitesse += force
            def maj_vitesse_agent_repulsion_mur(agent):
                for obstacle in self.obstacles:
                    force = f_repulsion_obstacle(agent, obstacle)
                    agent.vitesse += force
            #========================================================
            for agent in self.agents:
                if agent.alive:
                    agent.vitesse = np.array([0., 0.])
                    
                    maj_vitesse_agent_intention(agent)
                    maj_vitesse_agent_repulsion(agent)
                    maj_vitesse_agent_repulsion_mur(agent)
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
    
    if r < 2**(1/6)*sigma:
        
        return 4*epsilon*(-12*(sigma/r)**12/r+6*(sigma/r)**6/r)

    return 0
    
def fagent(agent1,agent2):
#Force de répulsion entre deux agents

    sigma=agent1.sigma
    epsilon=agent1.epsilon
    r = agent1.distance(agent2)
    
    vecteur_unitaire = (agent1.position - agent2.position) / r
    
    if agent2.alive:
        amplitude = -Dpotentiel(r,sigma*2,epsilon)
    else:
        amplitude = 0.
    
    return amplitude * vecteur_unitaire
    
def f_repulsion_obstacle(agent, obstacle):
    K, L = obstacle.sommets
    A = agent.position
    KL = L - K
    KA = A - K
    
    d = np.dot(KL, KA)
    theta = d / np.linalg.norm(KL)**2
    KH = theta * KL
    H = K + KH
    
    #print(theta)
    if theta < 0.:
        H = K
    elif theta > 1.:
        H = L
    #print(H)
    HA = -agent.position + H 
    #print(HA)
    distanceHA = np.linalg.norm(HA)
    HA_u = HA / distanceHA
    
    #print(distanceHA)
    potentiel = Dpotentiel(distanceHA, agent.sigma, agent.epsilon)
    force = HA_u * potentiel

    #print(potentiel)
    #print(force)
    return force
    
    
def random_agent(Lx, Ly, sigma, epsilon):
    return Agent(np.array([1 + 8 * random.random(), 1 + 13 * random.random()]), 1, sigma, epsilon)

def test_location(agent,zone):
    #la variable zone correspond à une liste de deux vecteurs qui permettent de former un carré (point angle en haut à gauche et point angle en bas à droite)
    #indique si l'agent est localisée dans la zone carrée
    lim_x_min = min(zone[0][0],zone[1][0])
    lim_x_max = max(zone[0][0],zone[1][0])
    lim_y_min = min(zone[0][1],zone[1][1])
    lim_y_max = max(zone[0][1],zone[1][1])    
    
    if(agent.position[0] >= lim_x_min and agent.position[0]<= lim_x_max):   
        if(agent.position[1] >= lim_y_min and agent.position[1]<= lim_y_max):
            return True
    return False
    

    




# Premier exemple

Lx = 20
Ly = 11.11
Nx = 400
Ny = 400

nombreT = 500
dt = T/nombreT
dt = 0.1
sigma = 0.1
epsilon = 1.0
T = 120




marie = Agent(np.array([5,5]), 1., sigma, epsilon*2, 'marie')
nirina = Agent(np.array([np.sqrt(2.)/2. * 5., np.sqrt(2.)/2. * 5.]), 1., sigma, epsilon/2, 'nirina')
luc = Agent(np.array([8., 2.]), 1., sigma, epsilon, 'luc')


#======================= test de la fonction test location =============================
test_zone = [[5.0,8.0],[10.0,6.0]]
agent_test = Agent(np.array([7.0,7.0]),1., sigma, epsilon, 'agent_1')
agent_test_2 = Agent(np.array([12.0,7.0]),1., sigma, epsilon, 'agent_2')
agent_test_3 = Agent(np.array([7.0,3.0]),1., sigma, epsilon, 'agent_3')
agent_test_4 = Agent(np.array([1.0,9.0]),1., sigma, epsilon, 'agent_4')
position_1 = test_location(agent_test,test_zone)
position_2 = test_location(agent_test_2,test_zone)
position_3 = test_location(agent_test_3,test_zone)
position_4 = test_location(agent_test_4,test_zone)
#=======================================================================================

# Murs d'exemple
largeur_porte = 2 * 0.90
#==============================================================================

murs = []
murs.append(Obstacle([[0,3],[0,11.11]]))
murs.append(Obstacle([[0,11.11], [13.52, 11.11]]))
murs.append(Obstacle([[13.52, 11.11], [13.52,3]]))

murs.append(Obstacle([[13.52 - largeur_porte, 3], [largeur_porte,3]]))
murs.append(Obstacle([[largeur_porte,4], [largeur_porte,3]]))
murs.append(Obstacle([[13.52 - largeur_porte, 4], [13.52 - largeur_porte,3]]))


murs.append(Obstacle([[13.52, 3 + 1], [13.52 - 0.25, 3 + 1]]))
murs.append(Obstacle([[13.52 - 0.25, 3 + 1], [13.52 - 0.25, 3]]))
murs.append(Obstacle([[13.52, 3], [13.52 - 0.25, 3]]))

murs.append(Obstacle([[0, 3 + 1], [0 + 0.25, 3 + 1]]))
murs.append(Obstacle([[0 + 0.25, 3 + 1], [0 + 0.25, 3]]))
murs.append(Obstacle([[0, 3], [0 + 0.25, 3]]))
 
table1_1 = Obstacle([[6, 10], [6, 11]]) 
table1_2 = Obstacle([[6, 11], [4, 11]])
table1_3 = Obstacle([[4, 11], [4, 10]])
table1_4 = Obstacle([[4, 10], [6, 10]])

def generer_table(debut, fin):
    x1, y1 = debut
    x2, y2 = fin
    table1_1 = Obstacle([[x1, y1], [x1, y2]]) 
    table1_2 = Obstacle([[x1, y2], [x2, y2]])
    table1_3 = Obstacle([[x2, y2], [x2, y1]])
    table1_4 = Obstacle([[x2, y1], [x1, y1]])
    return [table1_1, table1_2, table1_3, table1_4]
    
tables = []
for i in range(7):
    tables = tables + generer_table([1.5 + 0.25, 2.5 + 3 + i], [1.5 + 0.25 + 10, i + 2.9 + 3])
    



obstacles = murs + tables
#==============================================================================


porte = Porte([20,0], [20,3])


agents = [marie, nirina, luc]

for i in range(0):
    agents.append(random_agent(Lx, Ly, sigma, epsilon))
    
portes = [porte]
#obstacles=build_walls(Lx,Ly,portes)

salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)


fig, ax = plt.subplots(1,1)

fig.show()



salleTest.afficher(fig, ax) 

print(dt)
for i in range(nombreT):

    salleTest.maj()

    salleTest.afficher(fig, ax)
    #time.sleep(0.001)
    print(i)
    








