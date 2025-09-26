
# this file builds the database properly to be used
# by the training logic later on

import os
from PIL import Image
import numpy as np

x = []
y = []
for label in os.listdir("facesdatabase"):
    if os.listdir(f"facesdatabase/{label}"):
        for imgfile in os.listdir(f"facesdatabase/{label}"):
           img = Image.open(f"facesdatabase/{label}/{imgfile}").convert("RGB")
           x.append(np.array(img))
           y.append(int(label))

x = np.array(x)
y = np.array(y)
np.savez("database.npz", x=x, y=y)
print("Database built successfully.")
