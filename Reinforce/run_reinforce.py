from multiprocessing import Process, Pipe
from Reinforce.reinforce import Model
from Reinforce.runner import Runner
from Reinforce.Environment.catch import Catch
from Reinforce.policies import PolicyFullyConnected
from Reinforce.truncated_selection import select_top_worker
import time

class Worker:
    def __init__(self, id, performance):
        self.id = id
        self.performance = performance

def train_worker(worker_id, child):
    discount_rate = 0.99
    num_episodes = 10
    batch_size = 10
    env = Catch(5)

    policy = PolicyFullyConnected
    model = Model(policy, batch_size, 1, env.observation_space, env.action_space, 0.01)

    runner = Runner(env=env, discount_rate=discount_rate, num_episodes=num_episodes, batch_size=batch_size)
    time_steps = 1000

    while True:
        best_worker_id = child.recv() # continues while loop

        if best_worker_id is not None:
            model.load(best_worker_id) # load best model parameters
            model.explore()

        for i in range(0, time_steps):
            observations, rewards, actions = runner.step(model)
            model.train(observations, rewards, actions)
            performance = runner.average_episodic_performance

        model.save(worker_id)
        time.sleep(1)

        child.send((worker_id, performance)) # stop while loop

class Manager():

    def __init__(self, num_workers):
        self.population = []
        self.num_workers = num_workers

        self.first_call = True

        for worker_id in range(0, self.num_workers):
            worker = Worker(id=worker_id, performance=0.0)
            self.population.append(worker)

        self.start_processes()

    def start_processes(self):
        self.parents, childs = zip(*[Pipe() for _ in range(self.num_workers)])

        processes = []
        for i in range(0, self.num_workers):
            processes.append(
                Process(target=train_worker,
                        args=(i, childs[i])
                        ))

        # Start the process’s activity.
        for process in processes:
            process.start()

    def step(self):
        if self.first_call:
            for worker in self.population:
                self.parents[worker.id].send(None)
            self.first_call = False

        else:
            for worker in self.population:
                best_worker_id = select_top_worker(worker.id, self.population)
                if best_worker_id is not None:
                    print("Replace worker {} ({:.2f}) with worker {} ({:.2f})".format(worker.id,
                                                                                      self.population[
                                                                                          worker.id].performance,
                                                                                      best_worker_id,
                                                                                      self.population[
                                                                                          best_worker_id].performance))

                self.parents[worker.id].send(best_worker_id)

        # receive performance and update population
        for parent in self.parents:
            worker_id, performance = parent.recv()
            print("Worker {} performance {}".format(worker_id, performance))
            self.population[worker_id].performance = performance

def main():
    num_workers = 5
    manager = Manager(num_workers)

    time_steps = 100000
    for i in range(0, time_steps):
        manager.step()
        # worker are synchronized

if __name__ == '__main__':
   main()