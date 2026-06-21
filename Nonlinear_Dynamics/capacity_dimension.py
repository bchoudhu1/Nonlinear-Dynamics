#Count the number of boxes of size epsilon needed to cover a projection
#of the trajectory of a Lorenz attractor
#Do necessary imports 
import matplotlib.pyplot as plt 
import numpy as np

#Load in the data 
data = np.loadtxt('CapDimData.dat', delimiter=',', dtype=float)

# Extract x and z coordinates
x_data, z_data = data[:, 0], data[:, 2]  

#Plot data for validation
plt.plot(x_data, z_data)
plt.title("Lorenz Attractor (X-Z Projection)")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()

#Define parameter 
eps = 0.05

#Traj is a (n,2) array of a 2d projection of some dynamical system
#Count number of boxes to cover the trajectory
def count_boxes(eps, traj):
    traj = np.asarray(traj, dtype=float)
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)

    # Box index for each point
    box_ids = np.floor(traj / eps).astype(np.int64)

    # Count distinct boxes
    unique_boxes = np.unique(box_ids, axis=0)
    return len(unique_boxes)

#Create trajectory from data 
xz = data[:, [0, 2]]

#Count number of boxes 
nboxes = count_boxes(eps, xz)
print(nboxes)

# Use logarithmically spaced eps values
eps_vals = np.logspace(-2, 0, 50)   # 0.01 to 1

# Compute box counts
box_counts = np.array([count_boxes(eps, xz) for eps in eps_vals])

# Log quantities
log_one_eps = np.log(1/eps_vals)
log_box_counts = np.log(box_counts) 

# Indices defining scaling region (adjust after inspection)
i_start = 10
i_end   = 30

#Print capacity dimension 
slope, intercept = np.polyfit(
    log_one_eps[i_start:i_end+1],
    log_box_counts[i_start:i_end+1],
    1
)

print(f"Capacity Dimension = {slope:.4f}")

plt.figure(figsize=(8,6))

# Full data
plt.plot(log_one_eps, log_box_counts, 'ko-', ms=3)

# Scaling region boundaries
plt.axvline(log_one_eps[i_start], color='r', ls='--')
plt.axvline(log_one_eps[i_end], color='r', ls='--')

# Scaling region points
x_scale = log_one_eps[i_start:i_end+1]
y_scale = log_box_counts[i_start:i_end+1]

plt.plot(x_scale, y_scale, 'ro', label='Scaling region')

# Linear fit
y_fit = slope * x_scale + intercept
plt.plot(
    x_scale,
    y_fit,
    'b-',
    lw=3,
    label=fr'Fit: $D_c = {slope:.3f}$'
)

# Annotate dimension on plot
plt.text(
    0.05, 0.95,
    fr'$D_c = {slope:.3f}$',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.xlabel(r'$\log(1/\epsilon)$')
plt.ylabel(r'$\log N(\epsilon)$')
plt.title('Box-Counting Scaling Plot')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()