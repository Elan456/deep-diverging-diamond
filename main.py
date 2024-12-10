from simulation.simulation import Simulation


if __name__ == "__main__":
    sim = Simulation("./simulation/sim_defs/iso.csv", render=True)
    sim.run()
