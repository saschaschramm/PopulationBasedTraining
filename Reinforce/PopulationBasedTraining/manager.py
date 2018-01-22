from multiprocessing import Process, Pipe
from Reinforce.PopulationBasedTraining.truncated_selection import select_top_worker
from Reinforce.reinforce import Model
from Reinforce.runner import Runner
import time

class Worker:
    def __init__(self, id, performance):
        self.id = id
        self.performance = performance

class Manager():

    def __init__(self, env, runner_params, model_params):
        self.env = env
        self.population = []
        self.runner_params = runner_params
        self.model_params = model_params
        self.num_workers = self.runner_params["num_workers"]
        self.population = None
        self.start_processes()

    def start_processes(self):
        self.parents, childs = zip(*[Pipe() for _ in range(self.num_workers)])

        processes = []
        for worker_id in range(0, self.num_workers):
            processes.append(
                Process(target=self.train_worker,
                        args=(worker_id,
                              childs[worker_id],
                              self.runner_params,
                              self.model_params,
                              self.env)
                        ))

        # Start the process’s activity.
        for process in processes:
            process.start()


    def debug(self, worker_id, best_worker_id):
        print("Replace worker {} ({:.2f}) with worker {} ({:.2f})"
              .format(worker_id,
                      self.population[worker_id].performance,
                      best_worker_id,
                      self.population[best_worker_id].performance))

    def step(self):

        if self.population is None:
            self.population = []
            for worker_id in range(0, self.num_workers):
                worker = Worker(id=worker_id, performance=0.0)
                self.population.append(worker)
                self.parents[worker_id].send(None)
        else:
            for worker_id in range(0, self.num_workers):
                best_worker_id = select_top_worker(worker_id, self.population)
                if best_worker_id is not None:
                    self.debug(worker_id, best_worker_id)
                self.parents[worker_id].send(best_worker_id)


        # receive performance and update population
        for parent in self.parents:
            worker_id, performance = parent.recv()
            print("Worker {} performance {}".format(worker_id, performance))
            self.population[worker_id].performance = performance


    def train_worker(self, worker_id, child, runner_params, model_params, env):

        model = Model(model_params)
        runner = Runner(env=env,
                        model=model,
                        model_params = model_params,
                        runner_params=runner_params,
                        worker_id=worker_id,
                        )

        while True:
            best_worker_id = child.recv()  # continues while loop

            if best_worker_id is not None:
                model.load(best_worker_id)  # load best model parameters
                model.explore()

            for i in range(0, runner_params["ready_steps"]):
                observations, rewards, actions = runner.step()
                model.train(observations, rewards, actions)
                performance = runner.average_episodic_performance

            model.save(worker_id)
            time.sleep(1)
            child.send((worker_id, performance))  # stop while loop