import numpy as np

def fintention(agent, portes, Nx, Ny, grille = None):

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
    '''if grille != None:
        x, y = agent.position
        x = meter_to_int(x, Nx, Lx)
        y = meter_to_int(y, Ny, Ly)
        x_porte, y_porte = porte_cible.positionCentre
        x_porte = meter_to_int(x_porte, Nx, Lx)
        y_porte = meter_to_int(y_porte, Ny, Ly)
        try:
            path = bfs(Nx, Ny, grille, [x, y], [x_porte, y_porte])
            first_cell = np.array(path[0])
            vect = first_cell - agent.position
        except:
            1;
        
        '''
        
        
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
    