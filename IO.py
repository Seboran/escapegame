import matplotlib.pylab as plt
import matplotlib.animation as animation
from collections import defaultdict
import csv

def read_csv(filename, Lx, Ly):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        
        
        results = defaultdict(lambda : [])
        for row in spamreader:
            
            r, g, b, x, y, t = row

            color = ('#%02X%02X%02X' % (int(r), int(g), int(b)))
            results[int(t)].append([color, float(x), float(y)])
            # Print all data on a screen
        return results
    


def animate(title, salleTest, results, fig, axe, Nt):
    ''' This functions creates an animation and adds all the required legends
    Due to matplotlib limitations there are some trickled down technics to
    display a legend inside the code '''
    # Updates the screen
    def update(k):
        xs = []
        ys = []
        colors = []
        #print(k)
        axe.clear()
        axe.set_xlim(0, salleTest.Lx)
        axe.set_ylim(0, salleTest.Ly)
        axe.set_title(str(k))
        for agent_data in results[k]:
            color, x, y = agent_data
            xs.append(x)
            ys.append(y)
            colors.append(color)
        #print(xs)
        axe.scatter(xs, ys, c = colors)
        
        

    
        
    ''' Figure initialisation '''
    
    axe.set_xlim(0, salleTest.Lx)
    axe.set_ylim(0, salleTest.Ly)
    

    
    ani = animation.FuncAnimation(fig, update, frames = Nt, interval = 30, blit = False, repeat = True)

    
    return ani





