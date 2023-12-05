import time
import pandas as pd
import numpy as np
import seaborn as sns

import os, psutil
process = psutil.Process()

for _ in range(10):
    np.random.seed(42)
    df1 = pd.DataFrame({'key': np.arange(100000), 'value': np.random.rand(100000)})

    start_time = time.time()
    sns.scatterplot(data=df1)
    end_time = time.time()

    # Display benchmark results
    print(end_time - start_time)

print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
