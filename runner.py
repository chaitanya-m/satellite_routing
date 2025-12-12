
def run(optimiser, simulator, budget: int):
    for _ in range(budget):
        x = optimiser.ask()           # choose design
        y = simulator.evaluate(x)     # score design
        optimiser.tell(x, y)          # update optimiser
