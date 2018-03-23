import time
import scipy
import random
import collections
import numpy as np
from utils import *
from forces import *
import matplotlib.pylab as plt
import matplotlib.animation as animation

<<<<<<< HEAD:programmePrincipal.py
from simulation_salle_233_235_deux_portes import *
from Fonctions_classes import *

plt.close("all")



numero_agent = 1
numero_porte = -1
seuil_porte = 0.1
        
#class Agent:
#    def __init__(self, positionBase, vitesseBase, sigma, epsilon, nom = '', color = ''):
#        self.vitesseBase = vitesseBase
#        self.sigma = sigma
#        self.epsilon = epsilon
#        self.vitesse = np.array([0, 0])
#
#        self.position = positionBase
#        r = lambda: random.randint(0,255)
#        if color == '':
#            self.color = ('#%02X%02X%02X' % (r(),r(),r()))
#        else:
#            self.color = color
#        self.alive = True
#        self.nom = nom
#        # note : si quelqu'un traverse une porte il peut toujours gêner la sortie d'une salle
#        
#        
#    def distance(self, element):
#        if type(element) == Agent:
#            return np.linalg.norm(self.position - element.position)
#        if type(element) == Porte:
#            return np.linalg.norm(self.position - element.positionCentre)
#
#        
#class Porte:
#    def __init__(self, positionGauche, positionDroite):
#        self.positionGauche = np.array(positionGauche)
#        self.positionDroite = np.array(positionDroite)
#
#        self.positionCentre= 0.5*(self.positionDroite+self.positionGauche)
#
#        
#class Obstacle:
#    def __init__(self, sommets):
#        # Sommets est un couple de couples
#        self.sommets = np.array(sommets)
#    def __del__(self):
#        del self.sommets
#
#        
#class Grille:
#    def __init__(self, Nx, Ny): 
#        self.tab = [[0]*Nx for _ in range(Ny)]
#    
#        
#class Environnement:
#    def __init__(self, Lx, Ly, Nx, Ny, dt, obstacles, agents, portes):
#        self.obstacles = obstacles
#        self.agents = agents
#        self.portes = portes
#        self.Lx = Lx
#        self.Ly = Ly
#        self.Nx = Nx
#        self.Ny = Ny
#        self.dt = dt
#        self.grille = Grille(Nx, Ny)
#        
#        def meter_to_int(x, N, L):
#            return int(x * N / L) # dégueulasse
#        
#        for obstacle in obstacles:
#            # Remplis la grille avec les murs
##            debut, fin = obstacle
##            x1, y1 = debut
##            x2, y2 = fin
#            # On rajoute à chaque case où il y a un mur un entier non nul
##            for x in range(meter_to_int(x1), meter_to_int(x2) + 1):
##                for y in range(meter_to_int(y1), meter_to_int(y2) + 1):
#            2;
#            
#        for agent in agents:
#            nx = meter_to_int(agent.position[0], Nx, Lx)
#            ny = meter_to_int(agent.position[1], Ny, Ly)
#            
#            #self.grille.tab[nx][ny] = numero_agent
#            
#        for porte in portes:
#            #Todo
#            
#            
#            2;
#            
#    
#    
#    
#    
#            
#    def maj(self):
#        #========================================================
#        def maj_vitesse_agents():
#            #========================================================
#            def maj_vitesse_agent_intention(agent):
#                force = fintention(agent, self.portes)
#                agent.vitesse += force
#            #========================================================
#            #========================================================
#            def maj_vitesse_agent_repulsion(agent):
#                for agent_i in self.agents:
#                    
#                    if agent_i != agent:
#                        
#                        force = fagent(agent, agent_i)
#                        agent.vitesse += force
#            def maj_vitesse_agent_repulsion_mur(agent):
#                for obstacle in self.obstacles:
#                    force = f_repulsion_obstacle(agent, obstacle)
#                    agent.vitesse += force
#            #========================================================
#            for agent in self.agents:
#                if agent.alive:
#                    agent.vitesse = np.array([0., 0.])
#                    
#                    maj_vitesse_agent_intention(agent)
#                    maj_vitesse_agent_repulsion(agent)
#                    maj_vitesse_agent_repulsion_mur(agent)
#        #========================================================
#        
#        def maj_position_agents():
#            for agent in self.agents:
#                if agent.alive:
#                    agent.position = agent.position + self.dt * agent.vitesse
#                
#        def maj_agents_alive():
#            for agent in self.agents:
#                for porte in self.portes:
#                    if agent.distance(porte) < seuil_porte:
#                        agent.alive = False
#
#        maj_vitesse_agents()
#        maj_position_agents()
#        maj_agents_alive()
#    
#        
#    def afficher(self, figure, axe):
#        x = []
#        y = []
#        color = []
#        axe.set_xlim(0, self.Lx)
#        axe.set_ylim(0, self.Ly)
#        for agent in self.agents:
#            x.append(agent.position[0])
#            y.append(agent.position[1])
#            color.append(agent.color)
#            
#        
#        axe.scatter(x, y, c = color)
#        for obstacle in self.obstacles:
#
#            x = []
#            y = []
#            for sommet in obstacle.sommets:
#                x.append(sommet[0])
#                y.append(sommet[1])
#                
#            axe.plot(x, y, color = '#000000')
#            
#        for porte in self.portes:
#            pos_porte = porte.positionCentre
#            plt.plot(pos_porte[0], pos_porte[1], 'x')
#
#def build_walls(Lx,Ly,portes):
##On suppose que les portes sont bien définies
#    
#    liste_murs=[]
#    murd=[0,Ly]
#    murg=[0,Ly]
#    murh=[0,Lx]
#    murb=[0,Lx]
#    
#    for porte in portes:
#        
#        #Si x est le même : porte à gauche ou à droite
#        if porte.positionGauche[0]==porte.positionDroite[0]:
#            
#            xp=porte.positionGauche[0]
#            
#            #La porte est à gauche
#            if xp==0:
#                murg+=[porte.positionGauche[1],porte.positionDroite[1]]
#            #La porte est à droite
#            else:
#                murd+=[porte.positionGauche[1],porte.positionDroite[1]]
#            
#        #Sinon y est le même : porte en haut ou en bas
#        else:
#            
#            yp=porte.positionGauche[1]
#            
#            #La porte est en bas
#            if yp==0:
#                murb+=[porte.positionGauche[0],porte.positionDroite[0]]
#            #La porte est en haut
#            else:
#                murh+=[porte.positionGauche[0],porte.positionDroite[0]]
#    
#    murb=sorted(murb)
#    murh=sorted(murh)
#    murd=sorted(murd)
#    murg=sorted(murg)
#    
#    
#    #Build left walls
#    liste_murs+=[Obstacle([[0,murg[0]],[0,murg[1]]])]
#    
#    for i in range(2,len(murg),2):
#        
#        liste_murs.append(Obstacle([[0,murg[i]],[0,murg[i+1]]]))
#       
#    #Build right walls
#    liste_murs.append(Obstacle([[0,murd[0]],[0,murd[1]]]))
#    
#    for i in range(2,len(murd),2):
#        
#        liste_murs.append(Obstacle([[Lx,murd[i]],[Lx,murd[i+1]]]))
#        
#    #Build bottom walls
#    liste_murs.append(Obstacle([[murb[0],0],[murb[1],0]]))
#    
#    for i in range(2,len(murb),2):
#        
#        liste_murs.append(Obstacle([[murb[i],0],[murb[i+1],0]]))
#    
#    #Build top walls
#    liste_murs.append(Obstacle([[murh[0],Ly],[murh[1],Ly]]))
#    
#    for i in range(2,len(murh),2):
#        
#        liste_murs.append(Obstacle([[murh[i],0],[murh[i+1],0]]))
#            
#    return liste_murs
#
#def bfs(Nx, Ny, grid, start, goal):
#    queue = collections.deque([[start]])
#    seen = set([start])
#    while queue:
#        path = queue.popleft()
#        x, y = path[-1]
#        if grid[y][x] == goal:
#            return path
#        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
#            if 0 <= x2 < Nx and 0 <= y2 < Ny and grid[y2][x2] != 1 and (x2, y2) not in seen:
#                queue.append(path + [(x2, y2)])
#                seen.add((x2, y2))
#    return seen
#
#
#
#def fintention(agent, portes):
#
## Intention naturellle pour un agent d'aller vers la porte la plus proche
#
#    # Utilise l'algorithme de dijkstra
#    vect=portes[0].positionCentre - agent.position
#    vect_norm = np.linalg.norm(vect)
#    
#    
#    for porte in portes[1:]:
#        
#        vect_test = porte.positionCentre - agent.position
#        vect_test_norm = np.linalg.norm(vect_test)
#        
#        
#        if vect_test_norm < vect_norm:
#            
#            vect = vect_test
#    
#    vect = vect / vect_norm
#    return agent.vitesseBase * np.array(vect)
#
#
#    
#def Dpotentiel(r,sigma,epsilon):
##La dérivée du potentiel de répulsion
#    
#    if r < 2**(1/6)*sigma:
#        
#        return 4*epsilon*(-12*(sigma/r)**12/r+6*(sigma/r)**6/r)
#
#    return 0
#    
#def fagent(agent1,agent2):
##Force de répulsion entre deux agents
#
#    sigma=agent1.sigma
#    epsilon=agent1.epsilon
#    r = agent1.distance(agent2)
#    
#    vecteur_unitaire = (agent1.position - agent2.position) / r
#    
#    if agent2.alive:
#        amplitude = -Dpotentiel(r,sigma*2,epsilon)
#    else:
#        amplitude = 0.
#    
#    return amplitude * vecteur_unitaire
#    
#def f_repulsion_obstacle(agent, obstacle):
#    K, L = obstacle.sommets
#    A = agent.position
#    KL = L - K
#    KA = A - K
#    
#    d = np.dot(KL, KA)
#    theta = d / np.linalg.norm(KL)**2
#    KH = theta * KL
#    H = K + KH
#    
#    #print(theta)
#    if theta < 0.:
#        H = K
#    elif theta > 1.:
#        H = L
#    #print(H)
#    HA = -agent.position + H 
#    #print(HA)
#    distanceHA = np.linalg.norm(HA)
#    HA_u = HA / distanceHA
#    
#    #print(distanceHA)
#    potentiel = Dpotentiel(distanceHA, agent.sigma, agent.epsilon)
#    force = HA_u * potentiel
#
#    #print(potentiel)
#    #print(force)
#    return force
#    
#    
#def random_agent(Lx, Ly, sigma, epsilon):
#    return Agent(np.array([1 + 8 * random.random(), 1 + 13 * random.random()]), 1, sigma, epsilon)
#
#def test_location(agent,zone):
#    #la variable zone correspond à une liste de deux vecteurs qui permettent de former un carré (point angle en haut à gauche et point angle en bas à droite)
#    #indique si l'agent est localisée dans la zone carrée
#    lim_x_min = min(zone[0][0],zone[1][0])
#    lim_x_max = max(zone[0][0],zone[1][0])
#    lim_y_min = min(zone[0][1],zone[1][1])
#    lim_y_max = max(zone[0][1],zone[1][1])    
#    
#    if(agent.position[0] >= lim_x_min and agent.position[0]<= lim_x_max):   
#        if(agent.position[1] >= lim_y_min and agent.position[1]<= lim_y_max):
#            return True
#    return False
#    
#def agents_in_zone_count(agents,zone):
#    #fonction qui permet de compter combien d'agents sont présents dans la zone parmis tous ceux de la liste
#    nb_agent_in_zone = 0
#    for agent in agents:
#        in_zone = test_location(agent,zone)
#        if(in_zone == True):
#            nb_agent_in_zone += 1
#    return nb_agent_in_zone    
#
#
#def points_list(sommet_1, sommet_2):
#    pt_list = []
#    if(sommet_1[0] == sommet_2[0]):
#        #les deux points sont alignés verticalement
#        maxi = max(sommet_1[1],sommet_2[1])
#        mini = min(sommet_1[1],sommet_2[1])
#        diff = maxi - mini
#        pt_list.append([sommet_1[0], mini])
#        for i in range(1,diff):
#            tmp = [sommet_1[0],mini + i]
#            pt_list.append(tmp)
#        pt_list.append([sommet_1[0],maxi])
#        
#    elif(sommet_1[1] == sommet_2[1]):
#        #les deux points sont alignés horizontalemen
#        maxi = max(sommet_1[0],sommet_2[0])
#        mini = min(sommet_1[0],sommet_2[0])
#        diff = maxi - mini
#        pt_list.append([sommet_1[1], mini])
#        for i in range(1,diff):
#            tmp = [sommet_1[1],mini + i]
#            pt_list.append(tmp)
#        pt_list.append([sommet_1[1],maxi])
#    return pt_list
#    





# Premier exemple

Lx = 20
Ly = 11.11
Nx = 400
Ny = 400

nombreT = 500
dt = 0.01
sigma = 0.1
epsilon = 1.0
T = 120




marie = Agent(np.array([5,5]), 1., sigma, epsilon*2, 'marie')
nirina = Agent(np.array([np.sqrt(2.)/2. * 5., np.sqrt(2.)/2. * 5.]), 1., sigma, epsilon/2, 'nirina')
luc = Agent(np.array([8., 2.]), 1., sigma, epsilon, 'luc')


# Murs d'exemple
largeur_porte = 2 * 0.90
#==============================================================================

#Modélisation de l'aphithéâtre
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

#modélisation de la salle 233/235
largeur_porte = 0.8 #largeur porte de la grande salle
x0 = 1.0
y0 = 3.0
#murs_grd_salle = []
#murs_grd_salle.append(Obstacle([[x0,y0],[x0+0.1,y0]]))
#murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte,y0],[x0+0.1+largeur_porte+5.8,y0]]))
#murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0]]))
#murs_grd_salle.append(Obstacle([[x0+0.1+5.8+2*largeur_porte+4.2,y0],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
#murs_grd_salle.append(Obstacle([[x0,y0],[x0,y0+5.8]]))
#murs_grd_salle.append(Obstacle([[x0,y0+5.8],[x0+0.1+5.8+2*largeur_porte+4.2,y0+5.8]]))
#murs_grd_salle.append(Obstacle([[x0+0.1,y0],[x0+0.1,y0+0.8]]))
#murs_grd_salle.append(Obstacle([[x0+0.1+largeur_porte+5.8,y0],[x0+0.1+largeur_porte+5.8,y0+0.8]]))
#murs_grd_salle.append(Obstacle([[x0+5.9,y0+5.8],[x0+5.9,y0+5.6]]))

porte_grd_salle_1 = Porte([x0+0.1,y0-1], [x0+0.1+largeur_porte,y0-1])
porte_grd_salle_2 = Porte([x0+0.1+largeur_porte+5.8,y0-1], [x0+0.1+5.8+2*largeur_porte,y0-1])


#ajout des tables
espace_x_porte = 0.7
espace_x = 0.85 #espace qui separe dans la direction x deux tables
espace_y = 1.1 #espace qui separe dans la direction y deux tables
largeur_bureau_Gs = 0.5
longueur_bureau_Gs = 1.25

Obstacles_Gd_salle = Salle_233_235_deux_portes(x0,y0,largeur_porte,espace_x_porte,espace_x,espace_y,largeur_bureau_Gs,longueur_bureau_Gs)
#tables_Gs = []

#ajout des tables dans la parte gauche de la salle
#simulation des tables collées au mur 
#for i in range(3):
#    x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau_Gs + espace_x)
#    x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau_Gs + i*(espace_x + largeur_bureau_Gs)
#    y1 = y0
#    y2 = y1 + longueur_bureau_Gs
#    tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
#                             Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
##simulation des tables au centre de la pièce
#for i in range(4):
#    x1 = x0+0.1+largeur_porte + espace_x_porte + i*(largeur_bureau_Gs + espace_x)
#    x2 = x0+0.1+largeur_porte + espace_x_porte + largeur_bureau_Gs + i*(espace_x + largeur_bureau_Gs)
#    y1 = y0 + longueur_bureau_Gs + espace_y
#    y2 = y1 + longueur_bureau_Gs
#    y3 = y2 + longueur_bureau_Gs
#    tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
#                             Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]]),
#                             Obstacle([[x1,y2],[x1,y3]]), Obstacle([[x1,y3],[x2,y3]]),
#                             Obstacle([[x2,y3],[x2,y2]]), Obstacle([[x2,y2],[x1,y2]])]
#
##ajout des tables dans la parte droite de la salle   
##simulation des tables collées au mur
#for i in range(3):
#    x1 = x0+0.1+5.8+2*largeur_porte+4.2
#    x2 = x1 - longueur_bureau_Gs
#    y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
#    y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
#    tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
#                             Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
#
##simulation des tables au centre de la pièce
#for i in range(3):
#    x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(2*longueur_bureau_Gs + espace_x)
#    x2 = x1 - longueur_bureau_Gs
#    y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
#    y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
#    tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
#                             Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]
#for i in range(3):
#    x1 = x0+0.1+5.8+2*largeur_porte+4.2 -(longueur_bureau_Gs + espace_x)
#    x2 = x1 - longueur_bureau_Gs
#    if(i == 2):
#        x1 = x0+0.1+5.8+2*largeur_porte+4.2 - longueur_bureau_Gs
#        x2 = x1 - longueur_bureau_Gs
#    y1 = y0 + espace_y + i*(largeur_bureau_Gs + espace_y)
#    y2 = y0 + espace_y + largeur_bureau_Gs +i*(largeur_bureau_Gs + espace_y)
#
#    tables_Gs = tables_Gs + [Obstacle([[x1,y1],[x1,y2]]), Obstacle([[x1,y2],[x2,y2]]),
#                             Obstacle([[x2,y2],[x2,y1]]), Obstacle([[x2,y1],[x1,y1]])]

#ajout des agents    
eleves_grd_salle = []  
espace_agent_table = 0.17

for j in [0.25,0.75]:#A chaque bureau, on peut mettre un maximum de 2 élèves
    #ajout des élèves dans la partie gauche de la salle (on place un agent à chaque place)
    for i in range(3): 
        x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau_Gs + i*(largeur_bureau_Gs + espace_x)
        eleve = Agent(np.array([x+espace_agent_table,y0+j*longueur_bureau_Gs]), 1., sigma, epsilon*2, 'eleve')
        eleves_grd_salle.append(eleve)
    for i in range(4):
         x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau_Gs + i*(largeur_bureau_Gs + espace_x)
         y = y0+longueur_bureau_Gs+espace_y+j*longueur_bureau_Gs 
         eleve = Agent(np.array([x+espace_agent_table,y]), 1., sigma, epsilon*2, 'eleve')
         eleves_grd_salle.append(eleve)
    for i in range(4):
         x = x0+0.1+largeur_porte+espace_x_porte + largeur_bureau_Gs + i*(largeur_bureau_Gs + espace_x)
         y = y0+longueur_bureau_Gs+espace_y+j*longueur_bureau_Gs + longueur_bureau_Gs 
         eleve = Agent(np.array([x+espace_agent_table,y]), 1., sigma, epsilon*2, 'eleve')
         eleves_grd_salle.append(eleve)
    #ajout des élèves dans la partie droite de la salle (on place un agent à chaque place)
    for i in range(3):
        x =  x0+0.1+5.8+2*largeur_porte+4.2 - j*longueur_bureau_Gs
        y =  y0 + espace_y + i*(largeur_bureau_Gs + espace_y) +largeur_bureau_Gs
        eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
        eleves_grd_salle.append(eleve)
    for i in range(3):
         x = x0+0.1+5.8+2*largeur_porte+4.2 - (espace_x+2*longueur_bureau_Gs) - j*longueur_bureau_Gs
         y =  y0 + espace_y + i*(largeur_bureau_Gs + espace_y) +largeur_bureau_Gs
         eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
         eleves_grd_salle.append(eleve)
    for i in range(3):
        x = x0+0.1+5.8+2*largeur_porte+4.2 - (espace_x+longueur_bureau_Gs) - j*longueur_bureau_Gs
        if(i == 2):
            x = x0+0.1+5.8+2*largeur_porte+4.2 - longueur_bureau_Gs - j*longueur_bureau_Gs
        y =  y0 + espace_y + i*(largeur_bureau_Gs + espace_y) +largeur_bureau_Gs
        eleve = Agent(np.array([x,y + espace_agent_table]), 1., sigma, epsilon*2, 'eleve')
        eleves_grd_salle.append(eleve)

        

         
                            



=======
try:
    from tqdm import tqdm as progress
except:
    print("Please install tqdm for loading bar display")
    def progress(range):
        return range
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
                force = fintention(agent, self.portes, self.Nx, self.Ny, self.grille.tab)
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




def meter_to_int(x, N, L):
    return int(x * N / L) # dégueulasse  

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
    
def agents_in_zone_count(agents,zone):
    #fonction qui permet de compter combien d'agents sont présents dans la zone parmis tous ceux de la liste
    nb_agent_in_zone = 0
    for agent in agents:
        in_zone = test_location(agent,zone)
        if(in_zone == True):
            nb_agent_in_zone += 1
    return nb_agent_in_zone    


def points_list(sommet_1, sommet_2):
    pt_list = []
    if(sommet_1[0] == sommet_2[0]):
        #les deux points sont alignés verticalement
        maxi = max(sommet_1[1],sommet_2[1])
        mini = min(sommet_1[1],sommet_2[1])
        diff = maxi - mini
        pt_list.append([sommet_1[0], mini])
        for i in range(1,diff):
            tmp = [sommet_1[0],mini + i]
            pt_list.append(tmp)
            
        pt_list.append([sommet_1[0],maxi])
        
        
    elif(sommet_1[1] == sommet_2[1]):
        #les deux points sont alignés horizontalement
        maxi = max(sommet_1[0],sommet_2[0])
        mini = min(sommet_1[0],sommet_2[0])
        diff = maxi - mini
        pt_list.append([mini, sommet_2[1]])
        for i in range(1,diff):
            tmp = [mini + i, sommet_1[1]]
            pt_list.append(tmp)
        pt_list.append([maxi, sommet_1[1]])
    return pt_list
    

>>>>>>> 333eb094abe8ed2e40ded3a8b02d2e1ed59d9688:utils.py
def generer_table(debut, fin):
    x1, y1 = debut
    x2, y2 = fin
    table1_1 = Obstacle([[x1, y1], [x1, y2]]) 
    table1_2 = Obstacle([[x1, y2], [x2, y2]])
    table1_3 = Obstacle([[x2, y2], [x2, y1]])
    table1_4 = Obstacle([[x2, y1], [x1, y1]])
    return [table1_1, table1_2, table1_3, table1_4]


def bfs(Nx, Ny, grid, start, goal):
    ''' Prend des coordonnés cases sans unités
    Ne pas mettre de floats ou de mètres'''
    queue = collections.deque([[start]])
    
<<<<<<< HEAD:programmePrincipal.py
portes = [porte1, porte2]
#obstacles=build_walls(Lx,Ly,portes)

salleTest = Environnement(Lx, Ly, Nx, Ny, dt, obstacles, agents, portes)



#obstables_grd_salle = murs_grd_salle + tables_Gs
portes_grd_salle = [porte_grd_salle_1,porte_grd_salle_2]

grande_salleTest = Environnement(14,10,Nx,Ny,dt,Obstacles_Gd_salle,eleves_grd_salle,portes_grd_salle)



fig, ax = plt.subplots(1,1)

fig.show()


grande_salleTest.afficher(fig,ax)

#salleTest.afficher(fig, ax) 

#print(dt)
#for i in range(nombreT):
#
#    salleTest.maj()
#
#    salleTest.afficher(fig, ax)
#    #time.sleep(0.001)
#    print(i)
#    








=======
    
    
    grid_seen = np.zeros([Nx, Ny])
    
    grid_seen[start[0], start[1]] = 1
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < Nx and 0 <= y2 < Ny and grid[x2][y2] != 1 and grid_seen[x2, y2] != 1:
                queue.append(path + [[x2, y2]])
                grid_seen[x2, y2] = 1
    raise Exception
>>>>>>> 333eb094abe8ed2e40ded3a8b02d2e1ed59d9688:utils.py
