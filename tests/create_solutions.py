import numpy as np
from choccy.solutions import Solutions


if __name__ == '__main__':
    sols = Solutions(decs=np.zeros((100, 1)), objs=np.zeros((99, 1)))
    print(sols)