import time
import numpy as np

import os, psutil

process = psutil.Process()


for _ in range(10):

    def numpy_sum(a, b):
        return np.array(a) + np.array(b)

    # Set the size of the arrays
    array_size = 10**7

    # Create two lists of random numbers
    a_list = [i for i in range(array_size)]
    b_list = [i for i in range(array_size)]

    # Create two NumPy arrays of random numbers
    a_array = np.array(a_list)
    b_array = np.array(b_list)

    # Benchmark NumPy array addition
    start_time = time.time()
    result_array = numpy_sum(a_array, b_array)
    end_time = time.time()
    print(end_time - start_time)

print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
