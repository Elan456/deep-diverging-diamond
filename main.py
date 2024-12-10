from simulation.simulation import Simulation


if __name__ == "__main__":
    sim = Simulation("./simulation/sim_defs/sim1.csv", render=True)
    sim.run()
