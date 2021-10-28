import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from scipy.linalg import solve_continuous_are
import random
import matplotlib.pyplot as plt
import time
import cvxpy as cp
from cvxpy import OSQP
import gym

class LQR():
    def __init__(self,model): 
        A,B =  model.get_lin()
        Q = model.Q#np.diag([5,5,1,1]) 
        R = model.R#np.diag([100,100])
        P = solve_continuous_are(A,B,Q,R)
        self.Klqr = np.linalg.inv(R) @ B.T @ P
    
    def dx(self): 
        return -self.Klqr.copy()
    
    def __call__(self,x, x_d): 
        x = x.reshape(-1,1)
        x_desire = x_d.reshape(-1,1)

        return (- self.Klqr @ (x - x_desire)).flatten()


class MPC():
    def __init__(self, env, T):
        self.model = gym.make(env.unwrapped.spec.id)
        self.num_states = env.num_states
        self.num_actions = env.num_actions

        self.T = T

        self.Q = self.model.Q
        self.R = self.model.R

        self.u_min = self.model.action_space.low
        self.u_max = self.model.action_space.high

        self.x_min = self.model.observation_space.low
        self.x_max = self.model.observation_space.high

    

        self.x = cp.Variable((self.num_states, self.T+1))
        self.u = cp.Variable((self.num_actions, self.T))
        A,B = self.model.get_lin()
        self.A_mat = cp.Parameter(A.shape)
        self.B_mat = cp.Parameter(B.shape)
        self.x_r = cp.Parameter(self.num_states)
        self.x_init = cp.Parameter(self.num_states)

        cost = 0
        constr = []
        for t in range(self.T):
            cost += cp.quad_form(self.x[:, t] - self.x_r, self.Q) + cp.quad_form(self.u[:, t], self.R)
            constr += [self.x[:,t+1] == self.A_mat@self.x[:,t] + self.B_mat@self.u[:,t]]
            constr += [self.x_min <= self.x[:,t], self.x[:,t] <= self.x_max]
            constr += [self.u_min <= self.u[:,t], self.u[:,t] <= self.u_max]
        # sums problem objectives and concatenates constraints.
        constr += [self.x[:,0] == self.x_init]
        cost += cp.quad_form(self.x[:, self.T], self.Q)
        self.problem = cp.Problem(cp.Minimize(cost), constr)

    def __call__(self, state, final_state):
        
        self.x_init.value = state

        self.x_r.value = final_state

        A,B = self.model.get_lin()
        A = A*self.model.dt + np.eye(A.shape[0])
        B = B*self.model.dt
        self.A_mat.value = A
        self.B_mat.value = B

        self.problem.solve()


        return np.array([self.u.value[0][0], self.u.value[1][0]])



class MPC_CDG:
    def __init__(self, env, horizon):
        self.env = gym.make(env.unwrapped.spec.id)
        self.horizon = horizon

        self.Q = self.env.Q
        self.R = self.env.R
        self.RI = 1./self.R
        self.dldx = grad(self.loss_func, 0)

        self.dt = self.env.dt
        self.u = np.zeros((horizon,1))
        
    def forward(self, x_t, u_traj):
        self.env.reset(x_t.flatten())
        curr_state = x_t.copy()
        traj = []
        loss = 0.0
        for t in range(self.horizon):
            traj.append(curr_state)
            
            loss += self.loss_func(curr_state.reshape(-1,1), u_traj[t].reshape(-1,1))
            curr_state, _, _, _ = self.env.step(u_traj[t])
            
        return traj, loss, curr_state
            

    def loss_func(self, xx, uu):
        return xx.T@self.Q@xx + uu.T@self.R@uu

    def backward(self, state_traj, u_traj):
        rho = np.array([0.0,200,0.0,0.0]).reshape(-1,1)
        result_u = np.zeros((self.horizon,1))


        
        for t in reversed(range(self.horizon)):
            curr_dldx = self.dldx(state_traj[t].reshape(-1,1), u_traj[t])
            

            A,B = self.env.get_lin(state_traj[t], u_traj[t])

            
            rho = rho - (- curr_dldx - A.T@rho) * self.dt
            du = -self.RI@B.T@rho

            result_u[t] = du[0]
            
        return result_u


    def __call__(self, state, init_step_size, beta, max_u):
        k = init_step_size
        state_traj, loss, last_state = self.forward(state, self.u)
        du_traj = self.backward(state_traj, self.u)

        temp_action_traj = self.u + du_traj * k

        _, J2u, _ = self.forward(state, temp_action_traj)
        
        last_J2u = loss
        while J2u < last_J2u:
            k = k * beta
            temp_action_traj = self.u + du_traj * k
            _, new_J2u, _ = self.forward(state, temp_action_traj)
            last_J2u = J2u
            J2u = new_J2u
        
        k = k / beta
        self.u = self.u + du_traj * k
        

        return np.clip(self.u, -max_u, max_u)[0]