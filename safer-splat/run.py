#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from tqdm import tqdm
import json

from splat.gsplat_utils import GSplatLoader
from cbf.cbf_utils import CBF
from dynamics.systems import DoubleIntegrator, double_integrator_dynamics

from coords import replica2gsplat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for the CBF and dynamics
alpha = 5.
beta = 1.
dt = 0.05

# Methods for the simulation
n = 100         # number of different configurations
n_steps = 500   # number of time steps

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

### ----------------- Possible Distance Types ----------------- ###
# method = 'ball-to-ellipsoid'
# method = 'ball-to-ball-squared'
# method = 'ball-to-pt-squared'
# method = 'mahalanobis'
# method = 'ball-to-ball'
### ----------------- Possible Distance Types ----------------- ###

for scene_name in ['room_0']:
    for method in ['ball-to-ellipsoid']:
        
        if scene_name == 'office_3':
            radius = 0.1
            path_to_gsplat = Path('outputs/office_3/office_3/splatfacto/2025-04-17_132156/config.yml')

        elif scene_name == 'office_4':
            radius = 0.1
            path_to_gsplat = Path('outputs/office_4/office_4/splatfacto/2025-04-17_140913/config.yml')

        elif scene_name == 'room_0':
            radius = 0.03  
            path_to_gsplat = Path('outputs/room_0/room_0/splatfacto/2025-04-17_224945/config.yml')

        elif scene_name == 'stonehenge':
            radius_z = 0.01
            radius = 0.015
            radius_config = 0.784/2
            mean_config = np.array([-0.08, -0.03, 0.05])
            path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

        elif scene_name == 'flight':
            radius_z = 0.06
            radius = 0.03
            radius_config = 0.545/2
            mean_config = np.array([0.19, 0.01, -0.02])
            path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

        print(f"Running {scene_name} with {method}")

        tnow = time.time()
        gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)
        
        # Save point cloud as numpy
        point_cloud = gsplat.means.cpu().numpy()
        os.makedirs('point_clouds', exist_ok=True)
        np.save(f'point_clouds/{scene_name}_points.npy', point_cloud)
        print(f"Point cloud saved for {scene_name}: {len(point_cloud)} points")

        dynamics = DoubleIntegrator(device=device, ndim=3)

        # # Create configurations in a circle
        # x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
        # x0 = x0 + mean_config

        # xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
        # xf = xf + mean_config

        pairs = np.loadtxt(f'../utils/path/{scene_name}.csv', delimiter=',', skiprows=1)
        starts_raw = pairs[:, :3]
        goals_raw = pairs[:, 3:6]
        x0 = np.array([replica2gsplat(p, scene_name) for p in starts_raw])
        xf = np.array([replica2gsplat(p, scene_name) for p in goals_raw])

        # Run simulation
        total_data = []

        for trial, (start, goal) in enumerate(zip(x0, xf)):

            # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
            x = torch.tensor(start).to(device).to(torch.float32)
            x = torch.cat([x, torch.zeros(3).to(device).to(torch.float32)])
            goal = torch.tensor(goal).to(device).to(torch.float32)
            goal = torch.cat([goal, torch.zeros(3).to(device).to(torch.float32)])

            traj = [x]
            times = [0]
            u_values = []
            u_des_values = []
            safety = []
            sucess = []
            feasible = []
            total_time = []

            cbf = CBF(gsplat, dynamics, alpha, beta, radius, distance_type=method)
            for i in tqdm(range(n_steps), desc=f"Simulating trajectory {trial}"):

                # Simple PD controller to track to goal
                vel_des = 5.0*(goal[:3] - x[:3])
                vel_des = torch.clamp(vel_des, -0.1, 0.1)

                # add d term
                vel_des = vel_des + 1.0*(goal[3:] - x[3:])
                # cap between -0.1 and 0.1
                u_des = 1.0*(vel_des - x[3:])
                # cap between -0.1 and 0.1
                u_des = torch.clamp(u_des, -0.1, 0.1)

                ### ----------------- Safety Filtering ----------------- ###
                tnow = time.time()
                torch.cuda.synchronize()
                u = cbf.solve_QP(x, u_des)
                torch.cuda.synchronize()
                ### ----------------- End of Safety Filtering ----------------- ###

                total_time.append(time.time() - tnow)

                # We end the trajectory if the solver fails (because we short-circuit the control input if it fails)
                if cbf.solver_success == False:
                    print("Solver failed")
                    sucess.append(False)
                    feasible.append(False)
                    break

                # Propagate dynamics
                x_ = x.clone()
                x = double_integrator_dynamics(x,u)*dt + x

                traj.append(x)
                times.append((i+1) * dt)
                u_values.append(u.cpu().numpy())
                u_des_values.append(u_des.cpu().numpy())

                # record ball-to-ellipsoid distances
                if scene_name == 'flight-low-res':
                    # Use high res gsplat for comparison
                    h, grad_h, hess_h, info = gsplat_high_res.query_distance(x, radius=radius, distance_type='ball-to-ellipsoid')
                else:
                    h, grad_h, hess_h, info = gsplat.query_distance(x, radius=radius, distance_type='ball-to-ellipsoid')
                # record min value of h
                safety.append(torch.min(h).item())

                # It's not moving
                if torch.norm(x - x_) < 0.001:
                    # If it's at the goal
                    if torch.norm(x_ - goal) < 0.2:
                        print("Reached Goal")
                        sucess.append(True)
                        feasible.append(True)
                    else:
                        sucess.append(False)
                        feasible.append(True)
                    break

                # If it times out but still moving, we consider this loosely a success. 
                if i >= n_steps - 1:
                    sucess.append(True)
                    feasible.append(True)

            traj = torch.stack(traj)
            u_values = np.array(u_values)
            u_des_values = np.array(u_des_values)

            data = {
            'traj': traj.cpu().numpy().tolist(),
            'u_out': u_values.tolist(),
            'u_des': u_des_values.tolist(),
            'time_step': times,
            'safety': safety,
            'sucess': sucess,
            'feasible': feasible,
            'total_time': total_time,
            'cbf_solve_time': cbf.times_cbf,
            'qp_solve_time': cbf.times_qp,
            'prune_time': cbf.times_prune,
            }

            total_data.append(data)

        # Save trajectory
        data = {
            'scene': scene_name,
            'method': method,
            'alpha': alpha,
            'beta': beta,
            'radius': radius,
            'dt': dt,
            'total_data': total_data,
        }

        # create directory if it doesn't exist
        os.makedirs('trajs', exist_ok=True)

        # write to the file
        with open(f'trajs/{scene_name}_{method}.json', 'w') as f:
            json.dump(data, f, indent=4)

# %%