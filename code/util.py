import numpy as np


def parse_maze_file(filename):
    """
    Parse a 2D maze description text file.

    Example file for a 4x3 maze (from the book
        Stuart Russell, Peter Norvig: Artificial Intelligence: A Modern Approach (3rd ed)):

        # S: start, X: inaccessible, E: exit with reward +1, F: exit with reward -1
        ...E
        .X.F
        S...
    """
    A = np.loadtxt(filename, dtype=str).tolist()
    A = np.array([list(s) for s in A])
    if A.ndim != 2:
        # cannot convert to 2D array
        raise RuntimeError('require rectangular maze shape (each row must be of same length)')
    # convention that [0, 0] entry is at lower left corner, and that x-coordinate comes first
    A = np.flipud(A).T

    locs = []
    start = []
    exitsP = []     # exits with +1 rewards
    exitsN = []     # exits with -1 rewards
    for y in range(A.shape[1]):
        for x in range(A.shape[0]):
            if A[x, y] == '.':
                # default field
                locs.append([x, y])
            elif A[x, y] == 'S':
                # start field
                if start != []:
                    raise RuntimeError('only a single start field allowed')
                start = [x, y]
                locs.append([x, y])
            elif A[x, y] == 'X':
                # inaccessible field
                pass
            elif A[x, y] == 'E':
                exitsP.append([x, y])
                locs.append([x, y])
            elif A[x, y] == 'F':
                exitsN.append([x, y])
                locs.append([x, y])
            else:
                raise RuntimeError('invalid symbol "{}" encountered'.format(A[x, y]))

    if start == []:
        raise RuntimeError('maze requires a start field')

    return (A.shape[0], A.shape[1], locs, start, exitsP, exitsN)
