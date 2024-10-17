import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import qmc
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid, NetworkGrid
from mesa.datacollection import DataCollector

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
    return np.mean(adopters)

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
    def __init__(self, N, f, topology, torus, moore, k, memory, mobility_rate, influencer_placement="even", k_clumps=1):
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
        elif influencer_placement == "clumps":
            influencer_indices = []
            for i in range(k_clumps):
                clump_size = self.num_influencers // k_clumps
                clump = random.sample(range(N), clump_size)
                influencer_indices.extend(clump)

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

def run_simulation(
        N=100,
        f=.08,
        topology="lattice",
        torus=False,
        moore=False,
        k=4,
        memory=5,
        influencer_placement="even",
        k_clumps=2,
        mobility_rate=.25,
        steps=1000,
        show_steps=True,
        show_grid=True,
        show_every_n=100,
        show_results=True
    ):
    model = SocialNormModel(N, f, topology, torus, moore, k, memory, mobility_rate, influencer_placement, k_clumps)

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

        if compute_adoption_ratio(model) == 1:
            print(f"Ubiquity at step {step+1}")
            break

    if show_grid:
        influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
        influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
        plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1, is_grid=is_grid)

    adoption_data = model.datacollector.get_model_vars_dataframe()
    if show_results:
        adoption_data.plot()
        plt.title("Adoption Ratio Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Adoption Ratio")
        plt.show()

    return adoption_data["Adoption Ratio"].iloc[-1]

def find_tipping_point(
        N=100,
        f=.25,
        topology="lattice",
        torus=False,
        moore=False,
        k=4,
        memory=5,
        mobility_rate=.25,
        steps=1000,
        num_samples=25
    ):
    increased = False
    decreased = False
    sample = 0
    f_vals = []
    a_vals = []
    while sample < num_samples and f > 0 and f <= .5:
        print(f"Sample {sample+1}: f = {round(100*f, 2)}%")
        final_adoption_rate = run_simulation(
            N=N,
            f=f,
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
        if final_adoption_rate >= .5:
            old_f = f
            f -= .01*final_adoption_rate
            decreased = True
        else:
            old_f = f
            f += .01*(1 - final_adoption_rate)
            increased = True
        if increased and decreased:
            f_vals.append(old_f)
            a_vals.append(final_adoption_rate)
            sample += 1
    plt.hist(f_vals, bins=10, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title("Distribution of Tipping Points")
    plt.xlabel("f")
    plt.ylabel("Frequency")
    plt.show()

    f_est = np.mean(f_vals)
    n_est = f_est * N
    print(f"Estimated tipping point = {round(100*f_est, 2)}% ({round(n_est, 2)}/{N} agents)")

if __name__ == "__main__":
    find_tipping_point(
        N=225,
        topology="small-world",
        memory=0
    )