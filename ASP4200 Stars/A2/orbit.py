import numpy as np
import matplotlib.pyplot as plt

class Orbit:

    def integrate(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7, method='rk4'):
        method = method.lower()
        if method not in ['rk4', 'leapfrog']:
            raise Exception('No Such Method')
        
        if method == 'rk4':
            return self.integrate_rk4(dt, n_step, e)
        elif method == 'leapfrog':
            return self.integrate_leapfrog(dt, n_step, e)
        
    
    def integrate_rk4(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7):
        t_end = n_step * dt
        
        result_t = []
        result_variables = np.zeros((0, 4), dtype=float)
        
        # Initial Conditions
        t = 0.0
        variables = np.array([
            1 - e,
            0.0, 
            0.0,
            np.sqrt((1+e)/(1-e))
        ])
        
        result_t.append(t)
        result_variables = np.vstack((result_variables, [variables]))

        while t <= t_end:
            k1 = self.rhs(t, variables)
            k2 = self.rhs(t + dt / 2, variables + dt * k1 / 2)
            k3 = self.rhs(t + dt / 2, variables + dt * k2 / 2)
            k4 = self.rhs(t + dt, variables + dt * k3)

            # Increment
            t += dt
            variables += 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt

            result_variables = np.vstack((result_variables, [variables]))

        return result_t, result_variables
        
    def integrate_leapfrog(self, dt: float = 0.01, n_step: float = 5000, e: float = 0.7):
        ...

    
    def rhs(self, t, variables):
        
        # u' = v
        # v' = - (G * M * u) / (u^2 + h^2)^(3/2)   
        # h' = k
        # k' = - (G * M * h) / (u^2 + h^2)^(3/2)
        
        G = 1
        M = 1

        r = (variables[0] ** 2 + variables[2] ** 2) ** 0.5

        f1 = variables[1]
        f2 = - (G * M * variables[0]) / (r ** 3)
        f3 = variables[3]
        f4 = - (G * M * variables[2]) / (r ** 3)
        
        return np.array([f1, f2, f3, f4])


orbit = Orbit()

for dt in [0.01, 0.05, 0.1]:
    t, variables = orbit.integrate(dt=dt, n_step=5000)
    variables = variables.T

    plt.scatter(variables[0], variables[2], s=0.3, label=f'dt={dt}')


plt.legend()

# Set limit so we dont see the flying of when dt = 0.1
plt.ylim((-1, 1))
plt.xlim((-2, 0.5))
plt.xlabel('x')
plt.ylabel('y')

plt.show()