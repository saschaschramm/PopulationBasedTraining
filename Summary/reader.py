from os.path import join, isfile
from Summary import matplot
from os import listdir
from Summary.summary_pb2 import Summary
import numpy as np

truncate = lambda a,i : a[0:i]

class FileReader():

    def __init__(self, log_dir):
        self.log_dir = join("../Logs/", log_dir)
        self.summary = []
        self.first_write = True

    def read(self, file_name):
        try:
            with open(join(self.log_dir, file_name), 'rb') as file:
                summary = Summary()
                summary.ParseFromString(file.read())
                return summary
        except OSError as e:
            print(e)

    def plot(self, save=False):
        filesnames = [file for file in listdir(self.log_dir) if isfile(join(self.log_dir, file))]

        x = []
        y = []
        num_workers = len(filesnames)

        for filename in filesnames:
            summary = self.read(filename)
            x.append([episode.id for episode in summary.episodes])
            y.append([episode.performance for episode in summary.episodes])

        lengths = [len(list) for list in x]

        # truncate array x
        for i in range(0, len(x)):
            x[i] = truncate(x[i], min(lengths))

        # truncate array y
        for i in range(0, len(y)):
            y[i] = truncate(y[i], min(lengths))

        # mean
        mean_y = np.add.reduce(y, 0) / num_workers

        matplot.plot(x[0], mean_y, save)