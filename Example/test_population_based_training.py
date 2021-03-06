from Example.model import Model
from Example.population_based_training import Worker, train, exploit, step, eval, explore

import numpy as np

def train_worker(id, population, child):
    model = Model()
    params_store = []
    params_store.append(population[id].params)

    for i in range(0, 64):
        params = population[id].params
        hyperparams = population[id].hyperparams
        params = step(model, params, hyperparams)
        performance = eval(model, params, hyperparams)
        params_store.append(params)

        if i % 8 == 0:
            _, best_params = exploit(hyperparams, params, performance, population)

            if not np.array_equal(best_params, params):
                """
                best_params != params means that we have found a better model.
                In this example we don't pass the hyperparameters of the best model to the explore function because we
                want to analyse the effect on both model separately.
                """

                hyperparams, params = explore(hyperparams, best_params, performance, population)
                performance = eval(model, params, hyperparams)

        population[id] = Worker(id=id, params=params, hyperparams=hyperparams, performance=performance)

    # Send parameters to the other end of the connection
    child.send(params_store)

def main():
    workers = [
        Worker(id=0, params=[0.9, 0.9], hyperparams=[1.0, 0.3], performance=0),
        Worker(id=1, params=[0.9, 0.9], hyperparams=[0.3, 1.0], performance=0)
    ]

    train(train_worker, workers, 'Population Based Training')

if __name__ == '__main__':
   main()