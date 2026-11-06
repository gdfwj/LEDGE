def run_simulation(experiment, mode='fast'):
    from .simulation import run_simulation as run
    return run(experiment, mode)
