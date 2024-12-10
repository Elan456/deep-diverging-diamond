from simulation.simulation import Simulation


if __name__ == "__main__":
    sim = Simulation("./simulation/sim_defs/all_crash.csv", render=True)
    sim.run()
