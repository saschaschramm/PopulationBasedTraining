import numpy as np
from matplotlib import pyplot

def plot(params_x, params_y, title):
    linspace_x = np.linspace(start=0, stop=1, num=100)
    linspace_y = np.linspace(start=0, stop=1, num=100)

    x, y = np.meshgrid(linspace_x, linspace_y)
    z = 1.2 - (x ** 2 + y ** 2)

    pyplot.title(title)
    pyplot.xlabel(r'$\theta_0$')
    pyplot.ylabel(r'$\theta_1$')

    pyplot.plot(params_x[0], params_y[0], '.', color='blue')
    pyplot.plot(params_x[1], params_y[1], '.', color='red')
    pyplot.contour(x, y, z, colors='lightgray')

    pyplot.show()
    #pyplot.savefig("{}{}".format(title.replace(" ", "_"), ".png"))


