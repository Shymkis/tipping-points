import csv
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import qmc
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid, NetworkGrid
from mesa.datacollection import DataCollector

import statistics

def plot_grid(graph, influencers, influenced, step, is_grid=True):
    """Visualize the grid or network and color influencers and non-influencers differently for each time step."""
    plt.figure(figsize=(6, 6))

    if is_grid:
        pos = {(x, y): (x, y) for x, y in graph.nodes()}
    else:
        # pos = nx.spring_layout(graph, seed=517)
        pos = nx.circular_layout(graph)

    color_map = ['red' if node in influencers else 'orange' if node in influenced else 'lightblue' for node in graph.nodes()]

    # Plot the network
    nx.draw(graph, pos, node_color=color_map, with_labels=False, node_size=100)
    plt.title(f"Network at Time Step {step}")
    plt.show()

def compute_adoption_ratio(model):
    """Calculate the percentage of agents who adopted the '1' norm."""
    adopters = [agent.state for agent in model.schedule.agents]
    ord_adopters = [agent.state for agent in model.schedule.agents if not agent.influencer]
    return np.array([np.mean(adopters), np.mean(ord_adopters)])

class SocialAgent(Agent):
    def __init__(self, unique_id, model, initial_state, influencer=False, memory=0):
        super().__init__(unique_id, model)
        self.state = initial_state  # Either 0 (non-adopter) or 1 (adopter)
        self.influencer = influencer
        self.mobility_rate = model.mobility_rate
        self.memory = memory
        self.past_neighbor_states = []

    def step(self):
        if self.influencer and random.random() < self.mobility_rate:
            self.move()

        if isinstance(self.model.grid, SingleGrid):
            neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)

        if not self.influencer:
            interaction_result = self.interact(neighbors)
            if interaction_result != .5:
                self.state = round(interaction_result)

        name = "Influencer" if self.influencer else "Influenced" if self.state else "Susceptible"

    def interact(self, neighbors):
        current_neighbor_states = [n.state for n in neighbors]
        if len(self.past_neighbor_states) > self.memory * len(neighbors):
            del self.past_neighbor_states[:len(neighbors)]
        self.past_neighbor_states.extend(current_neighbor_states)
        return sum(self.past_neighbor_states) / len(self.past_neighbor_states)

    def move(self):
        if isinstance(self.model.grid, SingleGrid):  # Grid topology
            neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)
            non_influencer_neighbors = [n for n in neighbors if not n.influencer]
            if non_influencer_neighbors:
                neighbor = random.choice(non_influencer_neighbors)
                self.model.grid.swap_pos(self, neighbor)
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            non_influencer_neighbors = [n for n in neighbors if not n.influencer]
            if non_influencer_neighbors:
                neighbor = random.choice(non_influencer_neighbors)
                self_pos, neighbor_pos = self.pos, neighbor.pos
                self.model.grid.move_agent(self, neighbor_pos)
                self.model.grid.move_agent(neighbor, self_pos)

class SocialNormModel(Model):
    def __init__(self, N, f, topology, torus, moore, k, memory, mobility_rate, influencer_placement="even"):
        super().__init__()
        self.num_agents = N
        self.num_influencers = round(f * N)
        self.topology = topology
        self.torus = torus
        self.moore = moore
        self.mobility_rate = mobility_rate

        if topology == "lattice":
            self.G = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
            self.grid = SingleGrid(int(np.sqrt(N)), int(np.sqrt(N)), torus=self.torus)
        elif topology == "small-world":
            self.G = nx.watts_strogatz_graph(N, k=k, p=0.1)
            self.grid = NetworkGrid(self.G)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        self.schedule = SimultaneousActivation(self)

        if influencer_placement == "even":
            influencer_indices = self.get_regular_intervals(N, self.num_influencers)
        elif influencer_placement == "random":
            influencer_indices = random.sample(range(N), self.num_influencers)

        for i in range(self.num_agents):
            is_influencer = i in influencer_indices
            agent = SocialAgent(i, self, int(is_influencer), is_influencer, memory)
            if isinstance(self.grid, SingleGrid):
                self.grid.place_agent(agent, (i // int(np.sqrt(N)), i % int(np.sqrt(N))))
            else:
                self.grid.place_agent(agent, i)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={"Adoption Ratio": compute_adoption_ratio}
        )

    def get_regular_intervals(self, total_agents, num_influencers):
        if self.topology == "lattice":
            sampler = qmc.Halton(d=2)
            sample = sampler.random(n=num_influencers)
            indices = np.round(sample * np.sqrt(total_agents))
            indices = indices[:, 0] * int(np.sqrt(total_agents)) + indices[:, 1]
            return indices
        elif self.topology == "small-world":
            sampler = qmc.Halton(d=1)
            sample = sampler.random(n=num_influencers)
            indices = np.round(sample * total_agents).astype(int)
            return indices.flatten()

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

def run(
        f=.25,
        N=225,
        topology="lattice",
        torus=False,
        moore=False,
        k=4,
        memory=5,
        influencer_placement="even",
        mobility_rate=.25,
        steps=1000,
        show_steps=False,
        show_grid=False,
        show_every_n=500,
        show_results=False
    ):
    model = SocialNormModel(N, f, topology, torus, moore, k, memory, mobility_rate, influencer_placement)
    if model.num_influencers == 0:
        print("No influencers, no diffusion")
        return np.zeros(2)

    is_grid = (topology == "lattice")

    # Run the model
    for step in range(steps):
        if show_grid and step % show_every_n == 0:
            influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
            influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
            plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1, is_grid=is_grid)

        if show_steps:
            print(f"Step {step+1}/{steps}")
        model.step()

        if compute_adoption_ratio(model)[0] == 1:
            print(f"Ubiquity at step {step+1}")
            break

    if compute_adoption_ratio(model)[0] != 1:
        print(f"No ubiquity, final adoption rate = {np.round(100*compute_adoption_ratio(model), 2)}%")

    if show_grid:
        influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
        influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
        plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1, is_grid=is_grid)

    adoption_data = model.datacollector.get_model_vars_dataframe()
    if show_results:
        adoption_data.plot(title=None, legend=None)
        plt.xlabel("Steps")
        plt.ylabel("Adoption Rate")
        plt.tight_layout()
        plt.show()

    return adoption_data["Adoption Ratio"].iloc[-1]

def find_tipping_point(
        N=225,
        f=.25,
        topology="lattice",
        torus=False,
        moore=False,
        k=4,
        memory=5,
        mobility_rate=.25,
        steps=1000,
        num_samples=25,
        search_radius=.01,
        beta=4
    ):
    increased = decreased = False
    samples = []
    while len(samples) < num_samples:
        print(f"Sample {len(samples)+1}: f = {round(100*f, 2)}%")
        final_adoption_rate = run(
            f=f,
            N=N,
            topology=topology,
            torus=torus,
            moore=moore,
            k=k,
            memory=memory,
            mobility_rate=mobility_rate,
            steps=steps,
            show_steps=False,
            show_grid=False,
            show_results=False
        )
        print(f"Final adoption rate = {round(100*final_adoption_rate, 2)}%")
        # Update f according to the final adoption rate
        f_new = f + search_radius - 2*search_radius*final_adoption_rate # Linear adjustment
        f_new = f - search_radius + (2*search_radius)/(1 + (final_adoption_rate/(1 - final_adoption_rate)))**beta # Logistic adjustment
        # Ensure f stays above 0
        if f_new <= 0:
            f_new = f/2
        # Check if f has increased or decreased
        diff = f_new - f
        if diff > 0:
            increased = True
        elif diff < 0:
            decreased = True
        # Keep samples once f has increased and decreased
        if increased and decreased:
            samples.append(f)
        # Update f for the next iteration
        f = f_new
    # Save the results to a CSV file
    # with open('steps.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows([[topology, steps] + samples])
    # Calculate the estimated tipping point
    f_est = np.mean(samples)
    f_ci = 1.96*np.std(samples, ddof=1) / np.sqrt(len(samples))
    n_est = f_est * N
    n_ci = f_ci * N
    print(f"Estimated tipping point = {round(100*f_est, 2)}% ± {round(100*f_ci, 2)} (({round(n_est, 2)} ± {round(n_ci, 2)})/{N} agents)")

def runforstats(
        f=.25,
        N=225,
        topology="lattice",
        torus=False,
        moore=False,
        k=4,
        memory=5,
        influencer_placement="even",
        mobility_rate=.25,
        steps=1000,
        show_steps=False,
        show_grid=False,
        show_every_n=500,
        show_results=False
    ):
    model = SocialNormModel(N, f, topology, torus, moore, k, memory, mobility_rate, influencer_placement)
    if model.num_influencers == 0:
        print("No influencers, no diffusion")
        return -1

    is_grid = (topology == "lattice")

    # Run the model
    for step in range(steps):
        if show_grid and step % show_every_n == 0:
            influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
            influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
            plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1, is_grid=is_grid)

        if show_steps:
            print(f"Step {step+1}/{steps}")
        model.step()

        if compute_adoption_ratio(model)[0] == 1:
            #print(f"Ubiquity at step {step+1}")
            return (step+1)
            break

    #if compute_adoption_ratio(model)[0] != 1:
        #print(f"No ubiquity, final adoption rate = {np.round(100*compute_adoption_ratio(model), 2)}%")

    if show_grid:
        influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
        influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
        plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1, is_grid=is_grid)

    adoption_data = model.datacollector.get_model_vars_dataframe()
    if show_results:
        adoption_data.plot(title=None, legend=None)
        plt.xlabel("Steps")
        plt.ylabel("Adoption Rate")
        plt.tight_layout()
        plt.show()

    return -1

if __name__ == "__main__":
    for testf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for testmem in [3, 5, 7, 9, 10, 15, 20, 25, 30]:
            for testmob in [0, 0.05]:
                data = []
                failrate = 0
                print("f=", testf, "memory=", testmem, "mobility_rate=", testmob)
                for x in range(50):
                    val = runforstats(f=testf, memory=testmem, mobility_rate=testmob, topology="lattice", torus="True", moore="False")
                    if (val >= 0):
                        data.append(val)
                    else:
                        failrate += 2
                if (failrate == 98):
                    print("success=", data[0], "failrate=", failrate)
                elif (failrate > 98):
                    print("failure, failrate=", failrate)
                else:
                    print("mean=", statistics.mean(data), "variance=", statistics.variance(data), "failrate=", failrate)