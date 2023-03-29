import numpy as np
import matplotlib.pyplot as plt

class LaneEmden:

    def integrate(self, n: float = 1.5, dxi: float = 1e-2, xi_end: float = 10):
        result_xi = [0.0]
        result_theta = [1.0]
        result_dtheta_dxi = [0.0]

        # Initial Conditions
        xi = 0.0
        theta = 1.0
        dtheta_dxi = 0.0

        while xi <= xi_end:
            k1 = self.rhs(n, xi, theta, dtheta_dxi)
            k2 = self.rhs(n, xi + dxi / 2, theta + dxi * k1[0] / 2, dtheta_dxi + dxi * k1[1] / 2)
            k3 = self.rhs(n, xi + dxi / 2, theta + dxi * k2[0] / 2, dtheta_dxi + dxi * k2[1] / 2)
            k4 = self.rhs(n, xi + dxi, theta + dxi * k3[0], dtheta_dxi + dxi * k3[1])

            # Increment
            xi += dxi
            theta += 1/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * dxi
            dtheta_dxi += 1/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * dxi

            result_xi.append(xi)
            result_theta.append(theta)
            result_dtheta_dxi.append(dtheta_dxi)

            if n % 1 != 0 and theta < 0:
                # Estimate xi when theta -> 0
                # dxi * dtheta_dxi = dtheta = 0 - theta
                dxi = -theta / dtheta_dxi
                result_xi.append(xi + dxi)
                result_theta.append(0)
                result_dtheta_dxi.append(dtheta_dxi)         # Approx the same
                break
        
        return result_xi, result_theta, result_dtheta_dxi


    
    def rhs(self, n, xi, theta, dtheta_dxi):
        
        # u' = v                    (f1: solve for theta)
        # v' = -u^n - 2 * v / xi    (f2: solve for dtheta_dxi)

        f1 = dtheta_dxi
        if xi == 0: f2 = - 1 / 3
        else: f2 = - theta ** n - 2 * dtheta_dxi / xi

        return f1, f2



le = LaneEmden()

for n in np.arange(0, 6, 0.5):
    xi, theta, dtheta_dxi = le.integrate(n)

    plt.plot(xi, theta, label=f'n: {n}')

plt.legend()
plt.xlabel('xi')
plt.ylabel('theta')

plt.show()