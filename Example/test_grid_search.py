from Example.model import Model
from Example.population_based_training import Worker, train, step, eval

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
        population[id] = Worker(id=id, params=params, hyperparams=hyperparams, performance=performance)

    # Send parameters to the other end of the connection
    child.send(params_store)

def main():
    workers = [
        Worker(id=0, params=[0.9, 0.9], hyperparams=[1.0, 0.0], performance=0),
        Worker(id=1, params=[0.9, 0.9], hyperparams=[0.0, 1.0], performance=0)
    ]

    train(train_worker, workers, 'Grid Search')

if __name__ == '__main__':
   main()