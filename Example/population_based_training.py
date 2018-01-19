from multiprocessing import Process, Pipe, Manager
from Example.meshgrid import plot
import random

class Worker:
    def __init__(self, id, params, hyperparams, performance):
        self.id = id
        self.params = params
        self.hyperparams = hyperparams
        self.performance = performance

def train(target, workers, title):
    parents, childs = zip(*[Pipe() for _ in range(2)])

    population = Manager().list()
    population.append(workers[0])
    population.append(workers[1])

    processes = []
    for i in range(0,2):
        processes.append(Process(target=target, args=(i, population, childs[i])))

    # Start the process’s activity.
    for process in processes:
        process.start()

    params_x = []
    params_y = []

    for i in range(0,2):
        params_store = parents[i].recv()
        params_x.append([params[0] for params in params_store])
        params_y.append([params[1] for params in params_store])

    plot(params_x, params_y, title)

    # Block the calling thread until the process whose join() method is called terminates
    for process in processes:
        process.join()

    print(population[0].performance)
    print(population[1].performance)

def exploit(hyperparams, params, performance, population):
    "Use the rest of population to find better solution"

    best_hyperparams = hyperparams
    best_params = params
    best_performance = performance

    for model in population:
        if model.performance > best_performance:
            best_params = model.params
            best_hyperparams = model.hyperparams
            best_performance = model.performance

    return best_hyperparams, best_params

def explore(hyperparams, params, performance, population):
    # each hyperparameter independently is randomly perturbed by a factor of 1.2 or 0.8
    factors = [0.8, 1.2]
    h0 = random.choice(factors) * hyperparams[0]
    h1 = random.choice(factors) * hyperparams[1]

    return [h0, h1], params

def step(model, params, hyperparams):
    # Each iteration does a step of gradient descent
    model.update_params(params)
    model.update_hyperparams(hyperparams)
    model.train()
    new_params = model.get_params()
    return new_params

def eval(model, params, hyperparams):
    model.update_params(params)
    model.update_hyperparams(hyperparams)
    return model.eval()