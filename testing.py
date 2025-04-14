import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# Manipulator Parameters
# ---------------------------
l1 = 1.0         # Length of link 1 (thigh)
l2 = 1.0         # Length of link 2 (shank)

# ---------------------------
# Cable Model Parameters (2 redundancy: 4 cables)
# ---------------------------
# Nominal cable lengths (arbitrary, since we focus on rates)
L1_base = 0.5   # for link 1 cables
L2_base = 0.5   # for link 2 cables
# Sensitivity constants (m per radian)
a1 = 1.0        # for Link 1
a2 = 1.0        # for Link 2

# ---------------------------
# Fixed Anchor Points for Cables (visualization)
# ---------------------------
# Link 1: two cables
anchor1_link1 = np.array([-1.5, 1.5])  # Anchor for Cable 1 (upper)
anchor2_link1 = np.array([ 1.5, 1.5])  # Anchor for Cable 2 (lower)
# Link 2: two cables
anchor1_link2 = np.array([-1.5, -1.5]) # Anchor for Cable 3 (upper)
anchor2_link2 = np.array([ 1.5, -1.5])  # Anchor for Cable 4 (lower)

# ---------------------------
# Time Parameters
# ---------------------------
T_total = 10.0     # total simulation time (seconds)
N = 500            # number of time steps
t_vals = np.linspace(0, T_total, N)
dt = t_vals[1] - t_vals[0]

# ---------------------------
# Desired End-Effector Trajectory (force x=y motion)
# ---------------------------
# Here we design the inverse-kinematics so that the foot moves along x=y.
def s_function(t):
    # s(t) varies between 0 and sqrt(2)
    return (np.sqrt(2)/2) * (1 + np.sin(t))

def compute_joint_angles(t):
    """
    For desired end-effector position on the line x = y for a 2R manipulator with l1=l2=1:
      cos(q2) = s(t)^2 - 1,
      and q1 = π/4 - q2/2.
    """
    s = s_function(t)
    cos_q2 = np.clip(s**2 - 1, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    q1 = np.pi/4 - q2/2
    return q1, q2

# ---------------------------
# Compute Joint Trajectories and Velocities
# ---------------------------
q1_arr = np.zeros(N)
q2_arr = np.zeros(N)
for i in range(N):
    q1_arr[i], q2_arr[i] = compute_joint_angles(t_vals[i])
dq1_dt = np.gradient(q1_arr, dt)
dq2_dt = np.gradient(q2_arr, dt)
dq_dt = np.vstack((dq1_dt, dq2_dt))  # shape (2, N)

# ---------------------------
# Compute Cable Lengths and Their Rates (2 redundancy: 4 cables)
# ---------------------------
# For Link 1:
l1_upper = L1_base + a1 * q1_arr   # Cable 1 on link1
l1_lower = L1_base - a1 * q1_arr   # Cable 2 on link1
# For Link 2:
l2_upper = L2_base + a2 * q2_arr   # Cable 3 on link2
l2_lower = L2_base - a2 * q2_arr   # Cable 4 on link2

# Their time derivatives (analytical)
dl1_upper_dt = a1 * dq1_dt
dl1_lower_dt = -a1 * dq1_dt
dl2_upper_dt = a2 * dq2_dt
dl2_lower_dt = -a2 * dq2_dt

# Stack actual cable velocities (4 x N)
dl_actual = np.vstack((dl1_upper_dt, dl1_lower_dt, dl2_upper_dt, dl2_lower_dt))

# ---------------------------
# Define the Cable-Length Jacobian (J_c) and Its Pseudoinverse
# ---------------------------
# For our model:
#   l1_upper = L1_base + a1*q1,  l1_lower = L1_base - a1*q1,
#   l2_upper = L2_base + a2*q2,  l2_lower = L2_base - a2*q2.
# Therefore, J_c (4x2) is:
J_c = np.array([[ a1,   0],
                [-a1,   0],
                [ 0,   a2],
                [ 0,  -a2]])
# Its pseudoinverse (2x4) is:
J_c_dagger = np.array([[1/(2*a1), -1/(2*a1), 0, 0],
                       [0, 0, 1/(2*a2), -1/(2*a2)]])

# ---------------------------
# Task-Space Jacobian for a 2R Planar Arm
# ---------------------------
def J_task(q1, q2):
    J = np.array([[-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
                  [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2)]])
    return J  # shape (2,2)

# ---------------------------
# Compute Task-Space Velocity and Predicted Cable Velocities
# ---------------------------
v_task = np.zeros((2, N))
dl_pred = np.zeros((4, N))
for i in range(N):
    J_t = J_task(q1_arr[i], q2_arr[i])
    dq = dq_dt[:, i]
    v = J_t @ dq  # End-effector velocity (2,)
    v_task[:, i] = v
    # Predicted cable velocity: dl = J_c * dq
    dl_pred[:, i] = J_c @ dq

# ---------------------------
# Forward Kinematics (for display)
# ---------------------------
def forward_kinematics(q1, q2):
    base = np.array([0, 0])
    joint1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    ee = joint1 + np.array([l2 * np.cos(q1+q2), l2 * np.sin(q1+q2)])
    return base, joint1, ee

# ---------------------------
# Cable Attachment Points (for visualization)
# ---------------------------
def attachment_points(q1, q2):
    # For Link 1: attachment point (midpoint)
    attach_link1 = np.array([0.5 * l1 * np.cos(q1), 0.5 * l1 * np.sin(q1)])
    # For Link 2: attachment point (midpoint)
    joint1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    attach_link2 = joint1 + 0.5 * np.array([l2 * np.cos(q1+q2), l2 * np.sin(q1+q2)])
    return attach_link1, attach_link2

# ---------------------------
# Set up Matplotlib Animation for 2-Redundancy Case
# ---------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("End-Effector x=y Motion (2 Redundancy: 4 Cables)")

# Plot elements: manipulator, cable lines, velocity arrow, and info text.
line_manip, = ax.plot([], [], 'o-', lw=3, color='blue', label='Manipulator')
# Cable lines for Link 1
cable_line1, = ax.plot([], [], 'r--', lw=1.5, label='Cable1 Link1')
cable_line2, = ax.plot([], [], 'r--', lw=1.5, label='Cable2 Link1')
# Cable lines for Link 2
cable_line3, = ax.plot([], [], 'g--', lw=1.5, label='Cable1 Link2')
cable_line4, = ax.plot([], [], 'g--', lw=1.5, label='Cable2 Link2')
vel_arrow = ax.quiver(0,0,0,0, angles='xy', scale_units='xy', scale=1, color='red')
info_text = ax.text(-1.8, 1.7, "", fontsize=8, color="black")

def init_anim():
    line_manip.set_data([], [])
    cable_line1.set_data([], [])
    cable_line2.set_data([], [])
    cable_line3.set_data([], [])
    cable_line4.set_data([], [])
    vel_arrow.set_UVC(0,0)
    info_text.set_text("")
    return line_manip, cable_line1, cable_line2, cable_line3, cable_line4, vel_arrow, info_text

def animate(i):
    # Current joint angles from trajectory
    q1_i = q1_arr[i]
    q2_i = q2_arr[i]
    
    # Compute forward kinematics
    base, joint1, ee = forward_kinematics(q1_i, q2_i)
    x_manip = [base[0], joint1[0], ee[0]]
    y_manip = [base[1], joint1[1], ee[1]]
    line_manip.set_data(x_manip, y_manip)
    
    # Compute cable attachment points for visualization
    attach_link1, attach_link2 = attachment_points(q1_i, q2_i)
    
    # Update cable lines:
    # For Link 1 cables: attach at attach_link1 to their fixed anchors.
    cable_line1.set_data([attach_link1[0], anchor1_link1[0]], [attach_link1[1], anchor1_link1[1]])
    cable_line2.set_data([attach_link1[0], anchor2_link1[0]], [attach_link1[1], anchor2_link1[1]])
    # For Link 2 cables: attach at attach_link2 to their fixed anchors.
    cable_line3.set_data([attach_link2[0], anchor1_link2[0]], [attach_link2[1], anchor1_link2[1]])
    cable_line4.set_data([attach_link2[0], anchor2_link2[0]], [attach_link2[1], anchor2_link2[1]])
    
    # Compute end-effector velocity using task Jacobian:
    J_t = J_task(q1_i, q2_i)
    dq = np.array([dq1_dt[i], dq2_dt[i]])
    v = J_t @ dq  # End-effector velocity (2,)
    
    # Set the velocity arrow at the end-effector
    vel_arrow.set_offsets([ee[0], ee[1]])
    vel_arrow.set_UVC(v[0], v[1])
    
    # Retrieve current cable velocities from our analytical computations
    rate1 = dl_actual[0, i]  # Cable1 on Link 1
    rate2 = dl_actual[1, i]  # Cable2 on Link 1
    rate3 = dl_actual[2, i]  # Cable1 on Link 2
    rate4 = dl_actual[3, i]  # Cable2 on Link 2
    v_norm = np.linalg.norm(v)
    
    # Update info text overlay with cable rates and end-effector speed
    info = (f"||V|| = {v_norm:.3f} m/s\n"
            f"Cable Rates:\n"
            f"  L1-C1: {rate1:.3f} m/s\n"
            f"  L1-C2: {rate2:.3f} m/s\n"
            f"  L2-C1: {rate3:.3f} m/s\n"
            f"  L2-C2: {rate4:.3f} m/s")
    info_text.set_text(info)
    
    return line_manip, cable_line1, cable_line2, cable_line3, cable_line4, vel_arrow, info_text

ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init_anim,
                              interval=20, blit=True)

ax.legend(loc='upper right')
ani.save('manip.gif', writer='pillow', fps=30)
plt.show()
