from matplotlib import pyplot

def plot(x, y, save):
    pyplot.plot(x, y)
    pyplot.xlabel("Steps")
    pyplot.ylabel("Score")
    pyplot.tight_layout()
    pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if save:
        pyplot.savefig("../Logs/summary.png")
    else:
        pyplot.show()