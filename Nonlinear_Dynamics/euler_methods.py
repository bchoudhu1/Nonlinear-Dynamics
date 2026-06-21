#Forward and backward Euler for the Simple Harmonic Oscillator
# We will use the relation [x' v'] = [v -(k/m)*x]

#Necessary imports
import matplotlib.pyplot as plt

#Define single step for forward-euler method 
def forward_euler_onestep(init_conds, del_t, k, m):
    x_primed = [init_conds[1],-(k/m)*init_conds[0]]
    update_var = [del_t*i for i in x_primed]
    x_new = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var)]
    return x_new 

#Print out result for choice of initial conditions, k,m and del_t
#print(forward_euler_onestep([-1,-2], 0.1, 2, 0.5))

#Compute at time t=0.5 taking steps of 0.1
tot_time = 0.5   #Define variable here 
i=0
x_old = [-1,-2]
k =2
m = 0.5
step = 0.1
total_time = 0.0
iters = int(tot_time / step) - 1 

while i<=iters:
   x_new = forward_euler_onestep(x_old, step, k, m) 
   x_old = x_new
   total_time+=step 
   i+=1 

#Print results at end of iteration 
print("New coordinates are : "+str(x_new))   
print("Total time elapsed : "+ str(total_time))


#Generate trajectory in state-space 
def traj_forward_euler(steps, x_old, step, k, m):
    x_vals = []
    times = []
    i=0 
    total_time = 0.0
    while i<=steps-1:
        x_new = forward_euler_onestep(x_old, step, k, m)
        x_vals.append(x_new)
        x_old = x_new 
        total_time += step
        times.append(total_time)
        i+=1 
    return x_vals, times

#Call the function for 200 steps 
steps = 200 
x_vals, times = traj_forward_euler(steps, x_old, step, k, m)

#Plot results
plt.title("Trajectory generated for "+str(steps)+" steps with Forward-Euler")
plt.plot(times, x_vals)
plt.xlabel('Time')
plt.ylabel('Positions')
plt.show()

#Define single step for backward-euler method 
def backward_euler_onestep(init_conds, del_t, k, m):
    forward_euler_res = forward_euler_onestep(init_conds, del_t, k, m)
    x_primed = [forward_euler_res[1],-(k/m)*forward_euler_res[0]]
    update_var = [del_t*i for i in x_primed]
    x_new = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var)]
    return x_new

#Print out result for choice of initial conditions, k,m and del_t
#print(backward_euler_onestep([-1,-2], 0.1, 2, 0.5))

tot_time = 0.5   #Define variable here 
i=0
x_old = [-1,-2]
k =2
m = 0.5
step = 0.1
total_time = 0.0
iters = int(tot_time / step) - 1 

while i<=iters:
   x_new = backward_euler_onestep(x_old, step, k, m) 
   x_old = x_new
   total_time+=step 
   i+=1 

#Generate trajectory in state-space 
def traj_backward_euler(steps, x_old, step, k, m):
    x_vals = []
    times = []
    i=0 
    total_time = 0.0
    while i<=steps-1:
        x_new = backward_euler_onestep(x_old, step, k, m)
        x_vals.append(x_new)
        x_old = x_new 
        total_time += step
        times.append(total_time)
        i+=1 
    return x_vals, times

#Call the function for 200 steps 
steps = 200 
x_vals, times = traj_forward_euler(steps, x_old, step, k, m)

#Plot results
plt.title("Trajectory generated for "+str(steps)+" steps with Backward-Euler")
plt.plot(times, x_vals)
plt.xlabel('Time')
plt.ylabel('Positions')
plt.show()


#Print results at end of iteration 
print("New coordinates are : "+str(x_new))   
print("Total time elapsed : "+ str(total_time))

# Parameters
x0 = [-1, -2]
k = 2
m = 0.5
dt = 0.1
steps = 50

# Forward Euler trajectory
x_fe = [x0]
x_old = x0

for _ in range(steps):
    x_new = forward_euler_onestep(x_old, dt, k, m)
    x_fe.append(x_new)
    x_old = x_new

# Backward Euler trajectory
x_be = [x0]
x_old = x0

for _ in range(steps):
    x_new = backward_euler_onestep(x_old, dt, k, m)
    x_be.append(x_new)
    x_old = x_new

# Extract position and velocity coordinates
x_fe_pos = [state[0] for state in x_fe]
x_fe_vel = [state[1] for state in x_fe]

x_be_pos = [state[0] for state in x_be]
x_be_vel = [state[1] for state in x_be]

# State-space plot
plt.figure(figsize=(6,6))
plt.title('Compare Forward and Backward Euler State-Space Plot (50 steps, Δt = 0.1)')
plt.plot(x_fe_pos, x_fe_vel, label='Forward Euler')
plt.plot(x_be_pos, x_be_vel, label='Backward Euler')
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()


#Third way : Take strides that are average of forward-Euler and backward-Euler steps
#Also known as the trapezoidal rule

def trapezoidal_onestep(init_conds, del_t, k, m):
    x_primed_fwd = [init_conds[1],-(k/m)*init_conds[0]]
    update_var_fwd = [del_t*i for i in x_primed_fwd]
    x_new_fwd = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var_fwd)]
    x_primed_back = [x_new_fwd[1],-(k/m)*x_new_fwd[0]]
    update_var_back = [del_t*i for i in x_primed_back]
    update_var_mean = [(upd_1+upd_2)/2 for upd_1, upd_2 in zip(update_var_fwd, update_var_back)]
    x_new_trap = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var_mean)]
    return x_new_trap
    
#Helper function for plotting trajectories

def traj_trapezoidal(steps, x_old, step, k, m):
    x_vals = []
    times = []
    i=0 
    total_time = 0.0
    while i<=steps-1:
        x_new = trapezoidal_onestep(x_old, step, k, m)
        x_vals.append(x_new)
        x_old = x_new 
        total_time += step
        times.append(total_time)
        i+=1 
    return x_vals, times

#Call the function for 200 steps 
steps = 200 
x_vals, times = traj_forward_euler(steps, x_old, step, k, m)

#Plot results
plt.title("Trajectory generated for "+str(steps)+" steps for Trapezoidal Rule")
plt.plot(times, x_vals)
plt.xlabel('Time')
plt.ylabel('Positions')
plt.show()







