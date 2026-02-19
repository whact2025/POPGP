import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
WIDTH = 10
HEIGHT = 10
N = WIDTH * HEIGHT
dim = 2**N  # This is too large for explicit matrices (2^100)
            # We must use a local tensor network or effective rules.
            # Let's define the "Cells" directly as the fundamental objects.

# Simulation Parameters
NUM_CELLS = 100
STEPS = 50
LEAKAGE_THRESHOLD = 0.3  # Survival threshold
REPLICATION_PROB = 0.05
MUTATION_RATE = 0.02

# --- 1. Define Cell State ---
# Each cell is a 2x2 density matrix (1 qubit effective)
# Represented by 3 real parameters (Bloch vector: rx, ry, rz)
# |rho> = 0.5 * (I + rx*Sx + ry*Sy + rz*Sz)
# Radius r <= 1. Purity = (1 + r^2)/2. Entropy approx 1 - r^2.

class Cell:
    def __init__(self, x, y, bloch=None):
        self.x = x
        self.y = y
        if bloch is None:
            # Random pure state on surface of sphere
            phi = np.random.uniform(0, 2*np.pi)
            theta = np.random.uniform(0, np.pi)
            self.rx = np.sin(theta) * np.cos(phi)
            self.ry = np.sin(theta) * np.sin(phi)
            self.rz = np.cos(theta)
        else:
            self.rx, self.ry, self.rz = bloch
            
    def purity(self):
        r2 = self.rx**2 + self.ry**2 + self.rz**2
        return (1 + r2) / 2.0
        
    def entropy(self):
        r = np.sqrt(self.rx**2 + self.ry**2 + self.rz**2)
        # Clamp for safety
        r = min(r, 0.9999)
        # Eigs are (1+r)/2, (1-r)/2
        p1 = (1+r)/2
        p2 = (1-r)/2
        return -p1*np.log(p1) - p2*np.log(p2)

# Initialize Grid
grid = [[None for _ in range(HEIGHT)] for _ in range(WIDTH)]
cells = []

# Seed with random cells
for _ in range(int(NUM_CELLS * 0.4)):
    x = np.random.randint(0, WIDTH)
    y = np.random.randint(0, HEIGHT)
    if grid[x][y] is None:
        c = Cell(x, y)
        grid[x][y] = c
        cells.append(c)

print(f"Initialized {len(cells)} cells.")

# --- 2. Define Dynamics (Selection Principle) ---
# Each cell interacts with neighbors (Heisenberg-like)
# This entangles them, increasing local entropy (reducing purity).
# Interaction Strength J=0.2.
# Cells decay if they leak too much info (Selection).
# Cells replicate if they stay pure (Success).

def interact(c1, c2, dt=0.1):
    # Model: Purity Decay + Alignment
    # Rate of decay depends on mismatch (non-parallel Bloch vectors)
    # dot = cos(theta)
    dot = c1.rx*c2.rx + c1.ry*c2.ry + c1.rz*c2.rz
    
    # Entropy production rate
    # dS/dt ~ (1 - dot^2) * J
    # We implement this as shrinking the Bloch vector.
    
    # If aligned (dot=1), no decay.
    # If orthogonal (dot=0), max decay.
    decay_rate = 0.3 * (1.0 - dot**2) * dt
    factor = 1.0 - decay_rate
    
    c1.rx *= factor
    c1.ry *= factor
    c1.rz *= factor
    
    c2.rx *= factor
    c2.ry *= factor
    c2.rz *= factor
    
    # Synchronization (Alignment force)
    # This represents the "Self-Organization" part.
    # Cells that align survive better.
    align_strength = 0.1 * dt
    
    # Simple vector average pull
    avg_x = 0.5 * (c1.rx + c2.rx)
    avg_y = 0.5 * (c1.ry + c2.ry)
    avg_z = 0.5 * (c1.rz + c2.rz)
    
    c1.rx += align_strength * (avg_x - c1.rx)
    c1.ry += align_strength * (avg_y - c1.ry)
    c1.rz += align_strength * (avg_z - c1.rz)
    
    c2.rx += align_strength * (avg_x - c2.rx)
    c2.ry += align_strength * (avg_y - c2.ry)
    c2.rz += align_strength * (avg_z - c2.rz)

# Rule 2: Selection (Death)
# Threshold for death (High Entropy = Low Purity)
# Max Entropy = log(2) = 0.69
LEAKAGE_THRESHOLD = 0.4 


# Rule 2: Selection (Death)
# If entropy > Threshold (or radius < Threshold), cell dies (dissolves into substrate).

# Rule 3: Reproduction (Life)
# If a cell is very stable (low entropy) and has space, it replicates.

# Rule 3: Cooling (Stabilization)
# Real matter sheds entropy via radiation.
# Without this, closed systems heat up until death.
# We implement a small probability to "cool" (reset to pure state).
COOLING_PROB = 0.02

history_entropy = []
history_count = []
frames = []

print("Running Simulation...")

for step in range(STEPS):
    # 1. Interaction Phase
    # Iterate over grid neighbors
    for x in range(WIDTH):
        for y in range(HEIGHT):
            c = grid[x][y]
            if c is None: continue
            
            # Check neighbors (Von Neumann neighborhood)
            neighbors = []
            if x+1 < WIDTH and grid[x+1][y]: neighbors.append(grid[x+1][y])
            if y+1 < HEIGHT and grid[x][y+1]: neighbors.append(grid[x][y+1])
            if x>0 and grid[x-1][y]: neighbors.append(grid[x-1][y])
            if y>0 and grid[x][y-1]: neighbors.append(grid[x][y-1])
            
            for n in neighbors:
                interact(c, n)

    # 2. Selection, Reproduction & Cooling Phase
    new_grid = [[grid[x][y] for y in range(HEIGHT)] for x in range(WIDTH)]
    current_cells = []
    avg_ent = 0
    
    for x in range(WIDTH):
        for y in range(HEIGHT):
            c = grid[x][y]
            if c is None: continue
            
            ent = c.entropy()
            
            # Selection: Die if too mixed (high leakage)
            if ent > LEAKAGE_THRESHOLD:
                new_grid[x][y] = None
                continue
            
            # Cooling: Spontaneous emission of entropy
            if np.random.random() < COOLING_PROB:
                # Reset to a pure state (direction preserved, magnitude restored)
                norm = np.sqrt(c.rx**2 + c.ry**2 + c.rz**2)
                if norm > 1e-6:
                    c.rx /= norm
                    c.ry /= norm
                    c.rz /= norm
                ent = 0.0 # Effectively zero after cooling
                
            # Reproduction: Replicate if very pure and lucky
            if ent < 0.1 and np.random.random() < REPLICATION_PROB:
                # Find empty neighbor
                neigh_coords = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                np.random.shuffle(neigh_coords)
                for nx, ny in neigh_coords:
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and new_grid[nx][ny] is None:
                        # Copy with mutation
                        rx = c.rx + np.random.normal(0, MUTATION_RATE)
                        ry = c.ry + np.random.normal(0, MUTATION_RATE)
                        rz = c.rz + np.random.normal(0, MUTATION_RATE)
                        # Normalize
                        norm = np.sqrt(rx*rx + ry*ry + rz*rz)
                        new_grid[nx][ny] = Cell(nx, ny, (rx/norm, ry/norm, rz/norm))
                        break
            
            # Cell survives
            if new_grid[x][y] is not None:
                current_cells.append(new_grid[x][y])
                avg_ent += ent

    grid = new_grid
    
    # Record stats
    count = len(current_cells)
    if count > 0:
        avg_ent /= count
    history_entropy.append(avg_ent)
    history_count.append(count)
    
    # Visualization Frame
    frame = np.zeros((WIDTH, HEIGHT))
    for c in current_cells:
        # Color by purity (1 = pure/bright, 0 = mixed/dark)
        frame[c.x, c.y] = c.purity()
    frames.append(frame)

# --- 3. Visualization ---
# Plot 1: Population Dynamics
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Population', color=color)
ax1.plot(history_count, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Avg Entropy (Leakage)', color=color)
ax2.plot(history_entropy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Emergent Stability with Radiative Cooling")
plt.tight_layout()
plt.savefig("src/ca_model_results/dynamics_cooling.png")
print("Dynamics plot saved to src/ca_model_results/dynamics_cooling.png")

# Animation of the Grid
fig_anim, ax_anim = plt.subplots()
im = ax_anim.imshow(frames[0], cmap='inferno', vmin=0, vmax=1)
plt.title("Grid Evolution (Cooling Enabled)")

def update(frame):
    if frame >= len(frames): return [im]
    im.set_array(frames[frame])
    return [im]

ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), blit=True)
ani.save('src/ca_model_results/evolution_cooling.gif', writer='pillow', fps=10)
print("Animation saved to src/ca_model_results/evolution_cooling.gif")
