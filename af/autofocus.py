import time
import sys
import os

from utils import *
from sharpness import *
    

def main():
    """
    Args:
        argv[1]: path of dataset with cache.txt.
        argv[2]: one of the sharpness functions in sharpness.py. variance absolute is default.
    """

    dataPath = sys.argv[1] 
    sharpness_fn = eval(sys.argv[2]) if len(sys.argv)==3 else var_abs

    testData = parsing(dataPath)
    start = time.time()
    data = get_sharpness(testData, sharpness_fn)
    printResult(testData, data)

    print(f'time = {time.time()-start:.4f} sec.')



if __name__ == '__main__':
    main()
