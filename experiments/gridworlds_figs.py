import pylab as pl
import numpy as np

def fourRoom(X,Y):
    Y2 = (int) (Y/2)
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
        maze[x][Y2] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
        maze[X2][(int) (Y2/2)] = 1.
        maze[X2][(int) (3*Y2/2)] = 1.
        maze[(int) (X2/2)][Y2] = 1.
        maze[(int) (3*X2/2)][Y2] = 1.
    maze[1][1] = 0.3
    maze[-2][-2] = 0.7
    return maze

def twoRoom(X,Y):
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
    maze[X2][ (int) (Y/2)] = 1.
    maze[1][1] = 0.3
    maze[-2][-2] = 0.7
    return maze

grid = fourRoom(7, 7)
#grid = twoRoom(9, 11)


pl.figure()
pl.imshow(grid, cmap='hot', interpolation='nearest')
pl.savefig('gridworld4room.pdf')
