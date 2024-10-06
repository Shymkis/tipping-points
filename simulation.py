import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mesa import Agent, Model
from mesa.time import BaseScheduler, SimultaneousActivation, RandomActivation
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
    def __init__(self, unique_id, model, initial_state, influencer=False):
        super().__init__(unique_id, model)
        self.state = initial_state  # Either 0 (non-adopter) or 1 (adopter)
        self.influencer = influencer
        self.mobility_rate = model.mobility_rate

    def step(self):
        # Influencers may move based on mobility rate
        if random.random() < self.mobility_rate:
            self.move()
            ### Don't move into another influencer
            ### Influencers are eager to move to greener pastures after stagnation or reaching a certain threshold
            ### Those who get influenced can also move to a different location
            ### Influencer interaction rate can increase

        # Interact with neighbors (binary or n-ary interaction)
        if not self.influencer:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)
            interaction_result = self.interact(neighbors)

            # Update state based on interactions
            self.state = int(interaction_result > (len(neighbors) / 2))

    def interact(self, neighbors):
        """Count how many neighbors are in the '1' state (adopted norm)"""
        count = 0
        for neighbor in neighbors:
            if self.model.grid.is_cell_empty(neighbor.pos):
                continue
            neighbor_agent = self.model.grid.get_cell_list_contents([neighbor.pos])[0]
            if neighbor_agent.state:
                count += 1
        return count

    def move(self):
        """Move to a random neighboring position (von Neumann, Moore, etc.)"""
        neighbor = random.choice(list(self.model.grid.get_neighbors(self.pos, moore=self.model.moore, include_center=False)))
        self.model.grid.swap_pos(self, neighbor)

# --- Model Class ---
class SocialNormModel(Model):
    def __init__(self, N, f, topology, moore, mobility_rate, interaction_type, influencer_placement, k_clumps=1):
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

        # Create agents
        for i in range(N):
            initial_state = 0 if i >= self.num_influencers else 1
            influencer = i < self.num_influencers
            agent = SocialAgent(i, self, initial_state, influencer)
            self.grid.place_agent(agent, (i // int(np.sqrt(N)), i % int(np.sqrt(N))))
            self.schedule.add(agent)

        # Data Collection
        self.datacollector = DataCollector(
            model_reporters={"Adoption Ratio": compute_adoption_ratio}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# --- Running the simulation ---
def run_simulation(
        N=100,
        f=.1,
        topology="lattice",
        moore=False,
        steps=100,
        mobility_rate=.1,
        interaction_type="binary",
        influencer_placement="random",
        show_grid=False
    ):
    model = SocialNormModel(N, f, topology, moore, mobility_rate, interaction_type, influencer_placement)

    # Run the model
    for step in range(steps):
        print(f"Step {step+1}/{steps}")

        if show_grid:
            # Plot the grid after each step
            influencer_nodes = [agent.pos for agent in model.schedule.agents if agent.influencer]
            influenced_nodes = [agent.pos for agent in model.schedule.agents if not agent.influencer and agent.state]
            plot_grid(model.G, influencer_nodes, influenced_nodes, step=step+1)

        # Execute a step in the model
        model.step()

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
    f=.25,
    topology="lattice",
    moore=True,
    steps=1000,
    mobility_rate=.05,
    show_grid=False
)
