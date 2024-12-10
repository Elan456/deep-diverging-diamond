from simulation.simulation import Simulation


if __name__ == "__main__":
    sim = Simulation(render=True)
    sim.set_scenario(inputFile="./simulation/sim_defs/all_crash.csv")
    sim.run()
