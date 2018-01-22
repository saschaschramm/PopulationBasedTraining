from Summary.summary_pb2 import Episode, Summary
from os.path import join
import os

class FileWriter():

    def __init__(self, log_dir):
        self.log_dir = join("../Logs/", log_dir)
        self.summary = []
        self.first_write = True

    def write(self, filename, summary):
        os.makedirs(os.path.dirname(join(self.log_dir, filename)), exist_ok=True)

        with open(join(self.log_dir, filename), "wb") as file:
            file.write(summary.SerializeToString())

    def read(self, file_name):
        summary = Summary()
        try:
            with open(join(self.log_dir, file_name), 'rb') as file:
                summary.ParseFromString(file.read())
                return summary

        except OSError:
            return summary

    def add_summary(self, worker_id, episode_id, performance):
        episode = Episode()
        episode.id = episode_id
        episode.performance = performance
        filename = "summary_{}.bin".format(worker_id)

        if self.first_write == True:
            summary = Summary()
            self.first_write = False
        else:
            summary = self.read(filename) #read(template.format(path, worker_id))

        summary.episodes.extend([episode])
        self.write(filename, summary)








