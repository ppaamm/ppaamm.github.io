"""
Demonstration code for classification with genetic programming for the course
Emergence in Complex Systems (Athens TPT09 - March 2016) - Pierre-Alexandre Murena
This demonstration is based on the DEAP library.

All parameters of the GP can be set by the user as arguments of the main function.
Three kinds of data are used: multi-normal, half-moons and cells. The dataset 
can be chosen by the user by commenting/uncommenting the data generation part.
"""



import operator
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_moons

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


###############################################################################
######################## Data creation + visualization ########################
###############################################################################


def visualize(hof, ngen=-1):
    func = gp.compile(gp.PrimitiveTree(hof[0]), pset)
    #####    PLOT    #####
    
    h = .05     # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    delta_x, delta_y = x_max - x_min, y_max - y_min
    x_min -= 0.1 * delta_x
    x_max += 0.1 * delta_x
    y_min -= 0.1 * delta_y
    y_max += 0.1 * delta_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    y_pred = np.array(func(xx,yy) >= 0, dtype='f')
    plt.figure()
    plt.clf()
    
    cmap = colors.ListedColormap(['palevioletred', 'lightblue'])
    bounds=[-1,0.5,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    plt.imshow(y_pred, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cmap, norm=norm,
               aspect='auto', origin='lower')
    
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'or')
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ob')
    
    if ngen >= 0:
        plt.title('Round %i' % ngen)


def generate_data(n, centroids, sigma=1., random_state=42):
    """Generate sample data

    Parameters
    ----------
    n : int
        The number of samples
    centroids : array, shape=(k, p)
        The centroids
    sigma : float
        The standard deviation in each class

    Returns
    -------
    X : array, shape=(n, p)
        The samples
    y : array, shape=(n,)
        The labels
    """
    rng = np.random.RandomState(random_state)
    k, p = centroids.shape
    X = np.empty((n, p))
    y = np.empty(n)
    for i in range(k):
        X[i::k] = centroids[i] + sigma * rng.randn(len(X[i::k]), p)
        y[i::k] = i
    order = rng.permutation(n)
    X = X[order]
    y = y[order]
    return X, y


def generateRoundData(n_in, n_out, center, radius, sigma=0.5, random_state=42):
    """Generate sample round data

    Parameters
    ----------
    n_in : int
        The number of samples in the kernel
    n_out : int
        The number of samples in the border
    center : array, shape=(p,)
        The center of the cell
    sigma : float
        The standard deviation of the kernel data

    Returns
    -------
    X : array, shape=(n, p)
        The samples
    y : array, shape=(n,)
        The labels
    """
    rng = np.random.RandomState(random_state)
    p = center.shape[0]
    n = n_in + n_out
    X = np.empty((n, p))
    Y = np.empty(n)
    
    for i in range(n_in):
        X[i,:] = center + sigma * rng.randn(1,2)
        Y[i] = 1
    
    for i in range(n_out):
        theta = 2 * np.pi * rng.rand(1)[0]
        X[n_in + i,:] = center + [radius * np.cos(theta), radius * np.sin(theta)]
        Y[n_in + i] = 0
    return X,Y



###############################################################################
############S############### Genetic programming part ##########################
###############################################################################



# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.exp, 1)
#pset.addEphemeralConstant("rand106", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, X, Y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    def f (x): return func(x[0], x[1])
    # Evaluate the classification error
    n_errors = 0
    n = len(Y)
    for i in range(n):
        y_pred = int(f(X[i,:]) > 0)
        n_errors = n_errors + (y_pred != Y[i])
    return n_errors / float(n),



toolbox.register("evaluate", evalSymbReg, X=X, Y=Y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main(pop_size=300, mutProba=0.5, cxProba=0.1, nGenerationsPerRound=10, nRounds=5, random_state=42):
    random.seed(random_state)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    for g in range(nRounds):
        pop, log = algorithms.eaSimple(pop, toolbox, mutProba, cxProba, nGenerationsPerRound, 
                                       halloffame=hof, verbose=False)
        visualize(hof, g)

    return pop, log, hof

###############################################################################
############################### DATA GENERATION ###############################
###############################################################################

#### To select the dataset, uncomment the corresponding lines


# Binormal: each class is distributed on a gaussian
centroids = np.array([[-1, 0], [1,0]])
X, Y = generate_data(50, centroids)

# Half-moons: the classes form two half-moons
#X, Y = make_moons(200, noise=0.25, random_state=42)

# Two cells: one class is the border of the cells, the other is the kernel
#X, Y = generateRoundData(20, 40, np.array([-2,0]), 2)
#X1, Y1 = generateRoundData(20, 40, np.array([2,0]), 2)
#X = np.vstack((X,X1))
#Y = np.hstack((Y,Y1))


if __name__ == "__main__":
    pop, log, hof = main()    