# función para crear distribución bivariada

def bivariate_normal(x,y,sx,sy,mx,my):
    """ 
    Distribución bivariate normal, sin correlación entre las dos variables
    Se sigue la definición dada en MathWorld
    http://mathworld.wolfram.com/BivariateNormalDistribution.html
    """
    import numpy as np
    
    rho = 0
    z = (x-mx)**2/sx**2 + (y-my)**2/sy**2
    
    P = 1/(2*np.pi*sx*sy)*np.exp(-z/2.0)

    return P
    

