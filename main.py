from simulation.simulation import Simulation
import random 

if __name__ == "__main__":
    sim = Simulation(render=True)

    cars = []
    for _ in range(100):
        cars.append((random.randint(0, 15), random.randint(0, 100)))
    sim.set_scenario(cars=cars)
    sim.run()
