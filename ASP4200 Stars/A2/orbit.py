import numpy as np
import matplotlib.pyplot as plt


G = 1
M = 1

class Orbit:

    def integrate(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7, method='rk4'):
        method = method.lower()
        if method not in ['rk4', 'leapfrog', 'leapfrog_kdk']:
            raise Exception('No Such Method')
        
        if method == 'rk4':
            return self.integrate_rk4(dt, n_step, e)
        elif method == 'leapfrog':
            return self.integrate_leapfrog(dt, n_step, e)
        elif method == 'leapfrog_kdk':
            return self.integrate_leapfrog_kdk(dt, n_step, e)
        
    
    def integrate_rk4(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7):
        t_end = n_step * dt
        
        result_t = []
        result_variables = np.zeros((0, 4), dtype=float)
        
        # Initial Conditions
        t = 0.0
        variables = np.array([  
            1 - e,                  # x
            0.0,                    # y
            0.0,                    # dx/dt
            np.sqrt((1+e)/(1-e))    # dy/dt
        ])
        
        result_t.append(t)
        result_variables = np.vstack((result_variables, [variables]))

        while t <= t_end:
            k1 = self.rhs_rk4(t, variables)
            k2 = self.rhs_rk4(t + dt / 2, variables + dt * k1 / 2)
            k3 = self.rhs_rk4(t + dt / 2, variables + dt * k2 / 2)
            k4 = self.rhs_rk4(t + dt, variables + dt * k3)

            # Increment
            t += dt
            variables += 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt

            result_t.append(t)
            result_variables = np.vstack((result_variables, [variables]))

        return result_t, result_variables
        
    def integrate_leapfrog(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7):
        t_end = dt * n_step
        
        result_t = []
        result_variables = np.zeros((0, 4), dtype=np.float64)
                        
        # Initial Conditions
        t = 0.0
        variables_xy = np.array([  
            1 - e,                  # x
            0.0,                    # y
        ])
        variables_dxy_dt = np.array([  
            0.0,                    # dx/dt
            np.sqrt((1+e)/(1-e))    # dy/dt
        ])
            
        result_t.append(t)
        result_variables = np.vstack((result_variables, [np.concatenate((variables_xy, variables_dxy_dt))]))
    
        while t < t_end:
            a_i0 = self.rhs_leapfrog(t, variables_xy, e)
            
            variables_xy += variables_dxy_dt * dt + a_i0 / 2 * dt**2
            
            a_i1 = self.rhs_leapfrog(t, variables_xy, e)
            
            variables_dxy_dt += (a_i0 + a_i1) * dt / 2
            
            t += dt
            
            result_t.append(t)
            result_variables = np.vstack((result_variables, [np.concatenate((variables_xy, variables_dxy_dt))]))
        
        # print(result_variables)
        return result_t, result_variables
    
    def integrate_leapfrog_kdk(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7):
        t_end = dt * n_step
        
        result_t = []
        result_variables = np.zeros((0, 4), dtype=np.float64)
                        
        # Initial Conditions
        t = 0.0
        variables_xy = np.array([  
            1 - e,                  # x
            0.0,                    # y
        ])
        variables_dxy_dt = np.array([  
            0.0,                    # dx/dt
            np.sqrt((1+e)/(1-e))    # dy/dt
        ])
        
        variables_dxy_dt_half = variables_dxy_dt + self.rhs_leapfrog(t, variables_xy, e) * dt / 2
    
        result_t.append(t)
        result_variables = np.vstack((result_variables, [np.concatenate((variables_xy, variables_dxy_dt))]))
    
        while t < t_end:
            # x_i+1 = x_i + v_i+1/2 * dt
            # y_i+1 = y_i + u_i+1/2 * dt
            variables_xy += variables_dxy_dt_half * dt
            
            a = self.rhs_leapfrog(t, variables_xy, e)
            variables_dxy_dt = variables_dxy_dt_half + a * dt / 2
            variables_dxy_dt_half = variables_dxy_dt + a * dt / 2
            
            t += dt
            
            result_t.append(t)
            result_variables = np.vstack((result_variables, [np.concatenate((variables_xy, variables_dxy_dt))]))
        
        # print(result_variables)
        return result_t, result_variables
    
        
    def rhs_rk4(self, t, variables):
        
        # u' = v
        # h' = k
        # v' = - (G * M * u) / (u^2 + h^2)^(3/2)   
        # k' = - (G * M * h) / (u^2 + h^2)^(3/2)
        

        r = (variables[0] ** 2 + variables[1] ** 2) ** 0.5

        f1 = variables[2]
        f2 = variables[3]
        f3 = - (G * M * variables[0]) / (r ** 3)
        f4 = - (G * M * variables[1]) / (r ** 3)
        
        return np.array([f1, f2, f3, f4])
    
    def rhs_leapfrog(self, t, variables_xy, e):
        
        r = np.sqrt(variables_xy[0]**2 + variables_xy[1]**2)
        a_i = - G * M * variables_xy / r**3
        
        return a_i


orbit = Orbit()
# method = 'leapfrog'
# method = 'leapfrog_kdk'
method = 'rk4'

# Plot trajectory
fig_traj = plt.figure()
ax_traj: plt.Axes = fig_traj.add_subplot()

# Plot Angular Momentum
# L = x dy/dt - y dx/dt
fig_ang = plt.figure()
ax_ang: plt.Axes = fig_ang.add_subplot()

# Plot Energy
# E = (dx/dt^2 + dy/dt^2) / 2  - 1 / r 
fig_energy = plt.figure()
ax_energy: plt.Axes = fig_energy.add_subplot()

dts = [1e-3, 0.01, 0.05, 0.1]
# dts = [1e-3]

for dt in dts:
    t, variables = orbit.integrate(dt=dt, n_step=5000, method=method)
    variables = variables.T

    angular_momentum = variables[0] * variables[2] + variables[1] * variables[3]
    energy = (np.power(variables[2], 2) + np.power(variables[3], 2)) / 2 - 1 / np.sqrt(np.power(variables[0], 2) + np.power(variables[1], 2))

    ax_traj.scatter(variables[0], variables[1], s=0.3, label=f'dt={dt}')
    ax_ang.plot(t, angular_momentum, label=f'dt={dt}')
    ax_energy.plot(t, energy, label=f'dt={dt}')





ax_traj.legend()
ax_traj.set_xlabel('x')
ax_traj.set_ylabel('y')

ax_ang.legend()
ax_ang.set_xlabel('t')
ax_ang.set_ylabel('L')

ax_energy.legend()
ax_energy.set_xlabel('t')
ax_energy.set_ylabel('E')

if method == 'rk4':
    # Set limit so we dont see the flying of when dt = 0.1
    ax_traj.set_ylim((-1, 1))
    ax_traj.set_xlim((-2, 0.5))
    
    ax_ang.set_ylim((-1, 1))
    ax_ang.set_xlim((0, 300))
    
    ax_energy.set_ylim((-0.55, -0.495))
    ax_energy.set_xlim((0, 300))
    


plt.show()