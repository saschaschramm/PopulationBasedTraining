import random

def worker_is_bottom(id, population_sorted, num_selected_worker):
    bottom_worker = population_sorted[0:num_selected_worker]
    for worker in bottom_worker:
        if worker.id == id:
            return True
    return False

def select_top_worker(id, population):
    population_sorted = sorted(population, key=lambda x: x.performance)
    num_workers = len(population)
    num_selected_worker = int(num_workers * 0.2)

    if worker_is_bottom(id, population_sorted, num_selected_worker):
        top_worker = population_sorted[-num_selected_worker:]
        return random.choice(top_worker).id
    else:
        return None