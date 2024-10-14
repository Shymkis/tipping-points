import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mesa import Agent, Model
from mesa.time import SimultaneousActivation, StagedActivation
from mesa.space import SingleGrid, NetworkGrid
from mesa.datacollection import DataCollector

def plot_grid(grid, influencers, influenced, step):
    """Visualize the grid and color influencers and non-influencers differently for each time step."""
    plt.figure(figsize=(6, 6))

    # Get grid positions
    # pos = {node: (node//10, node%10) for node in grid.nodes()}
    # pos = nx.circular_layout(grid)
    pos = {(x,y): (x, y) for x,y in grid.nodes()}

    # Color map for nodes: influencers vs non-influencers
    color_map = ['red' if node in influencers else 'orange' if node in influenced else 'lightblue' for node in grid.nodes()]

    # Plot the grid
    nx.draw(grid, pos, node_color=color_map, with_labels=False, node_size=100)
    plt.title(f"Grid at Time Step {step}")
    plt.show()

# --- Utility functions for tracking the tipping point ---
def compute_adoption_ratio(model):
    """Calculate the percentage of agents who adopted the '1' norm."""
    adopters = [agent.state for agent in model.schedule.agents]
    return np.mean(adopters)

# --- Agent Class ---
class SocialAgent(Agent):
    def __init__(self, unique_id, model, initial_state, influencer=False, memory=0):
        super().__init__(unique_id, model)
        self.state = initial_state  # Either 0 (non-adopter) or 1 (adopter)
        self.influencer = influencer
        self.mobility_rate = model.mobility_rate
        self.memory = memory
        self.n_neighbors = 8 if model.moore else 4
        self.past_neighbor_states = []

    def step(self):
        # Influencers may move based on mobility rate
        if self.influencer and random.random() < self.mobility_rate:
            self.move()

        # Interact with neighbors (binary or n-ary interaction)
        if not self.influencer:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)
            interaction_result = self.interact(neighbors)

            # Update state based on interactions
            self.state = round(interaction_result)

    def interact(self, neighbors):
        current_neighbor_states = [n.state for n in neighbors]
        if len(self.past_neighbor_states) > self.memory * self.n_neighbors:
            del self.past_neighbor_states[:self.n_neighbors]
        self.past_neighbor_states.extend(current_neighbor_states)
        return sum(self.past_neighbor_states) / len(self.past_neighbor_states)

    def move(self):
        ### Influencers are eager to move to greener pastures after stagnation or reaching a certain threshold
        ### Move only where you have not been before
        neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)
        non_influencer_neighbors = [n for n in neighbors if not n.influencer]
        if non_influencer_neighbors:
            neighbor = random.choice(neighbors)
            self.model.grid.swap_pos(self, neighbor)

# --- Model Class ---
class SocialNormModel(Model):
    def __init__(self, N, f, topology, moore, memory, mobility_rate, interaction_type, influencer_placement="even", k_clumps=1):
        super().__init__()
        self.num_agents = N
        self.num_influencers = int(f * N)
        self.moore = moore
        self.mobility_rate = mobility_rate
        self.interaction_type = interaction_type

        # Create network based on topology
        if topology == "lattice":
            self.G = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
        elif topology == "small-world":
            self.G = nx.watts_strogatz_graph(N, k=4, p=.1)
        elif topology == "scale-free":
            self.G = nx.barabasi_albert_graph(N, 3)
        elif topology == "clique":
            self.G = nx.complete_graph(N)

        # self.grid = NetworkGrid(self.G)
        self.grid = SingleGrid(int(np.sqrt(N)), int(np.sqrt(N)), torus=False)
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
            self.grid.place_agent(agent, (i // int(np.sqrt(N)), i % int(np.sqrt(N))))
            self.schedule.add(agent)

        # # Create agents
        # for i in range(N):
        #     initial_state = 0 if i >= self.num_influencers else 1
        #     influencer = i < self.num_influencers
        #     agent = SocialAgent(i, self, initial_state, influencer, memory)
        #     self.schedule.add(agent)

        # Data Collection
        self.datacollector = DataCollector(
            model_reporters={"Adoption Ratio": compute_adoption_ratio}
        )

    def get_regular_intervals(self, total_agents, num_influencers):
        """ Return indices of influencers spaced at regular intervals throughout the grid. """
        interval = total_agents // num_influencers
        return list(range(0, total_agents, interval))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# --- Running the simulation ---
def run_simulation(
        N=100,
        f=.1,
        topology="lattice",
        moore=False,
        interaction_type="binary",
        memory=5,
        influencer_placement="even",
        k_clumps=2,
        mobility_rate=.1,
        steps=100,
        show_every_n=10
    ):
    model = SocialNormModel(N, f, topology, moore, memory, mobility_rate, interaction_type, influencer_placement, k_clumps)

    # Run the model
    for step in range(steps):
        # Plot the grid at regular intervals
        if step % show_every_n == 0:
            influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
            influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
            plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1)
        # Execute a step in the model
        print(f"Step {step+1}/{steps}")
        model.step()
        if compute_adoption_ratio(model) == 1:
            print(f"Norm is ubiquitous at step {step+1}")
            break
    # Plot the grid at the end
    influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
    influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
    plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1)

    # Collect and plot results
    adoption_data = model.datacollector.get_model_vars_dataframe()
    adoption_data.plot()
    plt.title("Adoption Ratio Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Adoption Ratio")
    plt.show()

# Example usage
run_simulation(
    N=100,
    f=.13,
    topology="lattice",
    moore=False,
    interaction_type="binary",
    memory=10,
    influencer_placement="even",
    k_clumps=5,
    mobility_rate=.25,
    steps=1000,
    show_every_n=100
)
