import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import numpy as np

with open(sys.argv[1]) as f:
    for line in f:
        tmp = line.strip().split(",")
        fname = tmp[0]
        buf = tmp[1:]
        plt.imsave(
            "/var/www/html/codeofduty/digit2/{}.png".format(fname),
            np.array(buf).reshape(28, 28),
            cmap=cm.gray)
