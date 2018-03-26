# compatibility
from __future__ import division
from __future__ import print_function


import time
import scipy
import random
import collections
import numpy as np

from utils import *

class Agent:
    def __init__(self, positionBase, vitesseBase, sigma, epsilon, nom = '', color = ''):
        self.vitesseBase = vitesseBase
        self.sigma = sigma
        self.epsilon = epsilon
        self.vitesse = np.array([0, 0])
        self.numbers = []

        self.position = positionBase
        r = lambda: random.randint(0,255)
        if color == '':
            self.numbers = (r(),r(),r())
            self.color = ('#%02X%02X%02X' % self.numbers)
        else:
            self.color = color
        self.alive = True
        self.nom = nom
        # note : si quelqu'un traverse une porte il peut toujours gêner la sortie d'une salle
        
        
    def distance(self, element):
        if type(element) == Agent:
            return np.linalg.norm(self.position - element.position)
        if type(element) == Porte:
            return np.linalg.norm(self.position - element.positionCentre)

        
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


def generer_table(debut, fin):
    x1, y1 = debut
    x2, y2 = fin
    table1_1 = Obstacle([[x1, y1], [x1, y2]]) 
    table1_2 = Obstacle([[x1, y2], [x2, y2]])
    table1_3 = Obstacle([[x2, y2], [x2, y1]])
    table1_4 = Obstacle([[x2, y1], [x1, y1]])
    return [table1_1, table1_2, table1_3, table1_4]


        
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
        
        
        
        for obstacle in obstacles:
            # Remplis la grille avec les murs
            debut, fin = obstacle.sommets
            x1, y1 = debut
            x2, y2 = fin
            x1 = meter_to_int(x1, Nx, Lx)
            x2 = meter_to_int(x2, Nx, Lx)
            y1 = meter_to_int(y1, Ny, Ly)
            y2 = meter_to_int(y2, Ny, Ly)
            fill_mur = points_list([x1, y1], [x2, y2])
            
            for x, y in fill_mur:
                
                
                # On rajoute une case là où il y a un mur
                # Vérifier que la case ne quitte pas la grille
                if x >= 0 and x < Nx and y >= 0 and y < Ny:
                    self.grille.tab[x][y] = 1
           
            
        for agent in agents:
            nx = meter_to_int(agent.position[0], Nx, Lx)
            ny = meter_to_int(agent.position[1], Ny, Ly)
            
            #self.grille.tab[nx][ny] = numero_agent
            
        for porte in portes:
            #Todo
            
            
            2;
            
    
    
    
    
            
    def maj(self):
        #========================================================
        def maj_vitesse_agents():
            #========================================================
            def maj_vitesse_agent_intention(agent):
                force = fintention(agent, self.portes, self.Nx, self.Ny, self.Lx, self.Ly, self.grille.tab)
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
                
                if not(test_location(agent, [[0, 3], [13.52, 11.11]])):
                    agent.alive = False

        maj_vitesse_agents()
        maj_position_agents()
        maj_agents_alive()
        
    def export_agents(self, t):
        for agent in self.agents:
            yield [agent.numbers, agent.position, t]

    def maj_turns(self, N):
        
        yield list(self.export_agents(0))
        progress_bar = progress(range(N))
        for i in progress_bar:
            
            self.maj()
            
            nombre_agents = agents_in_zone_count(self.agents, [[0, 3], [13.52, 11.11]])
            test_location
            
            progress_bar.set_description(desc = "Reste " + str(nombre_agents) + " agents")
            yield list(self.export_agents(i + 1))
            
    
        
    def afficher(self, figure, axe):
        x = []
        y = []
        color = []
        axe.set_xlim(0, self.Lx)
        axe.set_ylim(0, self.Ly)
        '''for agent in self.agents:
            x.append(agent.position[0])
            y.append(agent.position[1])
            color.append(agent.color)
            
        
        axe.scatter(x, y, c = color)'''
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
def fintention(agent, portes, Nx, Ny, Lx, Ly, grille = None):

# Intention naturellle pour un agent d'aller vers la porte la plus proche

    # Utilise l'algorithme de dijkstra
    vect=portes[0].positionCentre - agent.position
    vect_norm = np.linalg.norm(vect)
    porte_cible = portes[0]
    
    for porte in portes[1:]:
        
        vect_test = porte.positionCentre - agent.position
        vect_test_norm = np.linalg.norm(vect_test)
        
        
        if vect_test_norm < vect_norm:
            
            vect = vect_test
            porte_cible = porte
    
    vect = vect / vect_norm
    '''Now we know where we're going to we know take the route to the destination
    and cut until the distance is too far'''
    
    # if grille != None:
    #     ''' Convert position and door to ints'''
    #     x, y = agent.position
    #     x = meter_to_int(x, Nx, Lx)
    #     y = meter_to_int(y, Ny, Ly)
    #     x_porte, y_porte = porte_cible.positionCentre
    #     x_porte = meter_to_int(x_porte, Nx, Lx)
    #     y_porte = meter_to_int(y_porte, Ny, Ly)
    #     ''' Now we get the shortest path '''
    #     try:
    #         path = bfs(Nx, Ny, grille, [x, y], [x_porte, y_porte])
    #         """ We can't to cut our shortest path until we reach max walking distance """
    #         distance_walked = 0
    #         destination = []
    #         k = 0
    #         while distance_walked < agent.vitesseBase:
    #             x_int, y_int = path[k]
    #             x = int_to_meter(x_int, Nx, Lx)
    #             y = int_to_meter(y_int, Ny, Ly)
    #             cell = np.array([x, y])
                
                
    #             vect = first_cell - agent.position
    #             distance_walked = np.linalg.norm(vect)
    #             k += 1
    #         return np.array(vect)
    #     except:
    #         1;
        
        
        
        
    return agent.vitesseBase * np.array(vect)


    
def Dpotentiel(r,sigma,epsilon, D_sat):
#La dérivée du potentiel de répulsion
    
    if r < 2**(1/6)*sigma:
        
        return min(4*epsilon*(-12*(sigma/r)**12/r+6*(sigma/r)**6/r), D_sat)

    return 0
    
def fagent(agent1,agent2):
#Force de répulsion entre deux agents

    sigma=agent1.sigma
    epsilon=agent1.epsilon
    r = agent1.distance(agent2)
    
    vecteur_unitaire = (agent1.position - agent2.position) / r
    
    if agent2.alive:
        amplitude = -Dpotentiel(r,sigma*2,epsilon, agent1.vitesseBase)
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
    potentiel = Dpotentiel(distanceHA, agent.sigma, agent.epsilon, agent.vitesseBase)
    force = HA_u * potentiel

    #print(potentiel)
    #print(force)
    return force
    