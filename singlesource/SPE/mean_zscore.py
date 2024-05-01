import os
import pandas as pd

file_path = "output_final_f1.csv"
import numpy as np
from itertools import combinations



# for distance in project_distances:
#     print(distance)
df = pd.read_csv(file_path).iloc[:,0:5]
projectlist = df.values.tolist()
distname = []
for line in projectlist:
        dist=[]
        target = line[0]
        sourcelist = line[1:]
        for source in sourcelist:
            str_t = target.split('-')[0]
            str_s = source.split('-')[0]
            if str_s!=str_t:
                    finalsource = source
                    break
        dist.append(target)
        dist.append(finalsource)
        distname.append(dist)
for i in distname:
    print(i)
df = pd.DataFrame(np.array(distname))
df.to_csv("sort_final_f1.csv",index=False)