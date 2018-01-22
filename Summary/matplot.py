from matplotlib import pyplot

def plot(x, y, save):
    pyplot.plot(x, y)
    pyplot.xlabel("Episodes")
    pyplot.ylabel("Performance")
    pyplot.tight_layout()

    if save:
        pyplot.savefig("../Logs/summary.png")
    else:
        pyplot.show()