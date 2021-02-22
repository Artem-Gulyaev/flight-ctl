#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import *
import matplotlib.pyplot as plt

class Pendulum:

    # RETURNS: default parameters for the system
    def default_params():
        return {
                "g_mps2": 9.81
                , "l_m": 1.0
                , "m_kg": 1.0
               }

    # RETURNS:
    #   the number of the dynamic generalized variables
    #   of the current system
    @property
    def dynamic_vars_N(self):
        return 1

    # RETURNS: default initial state for the system
    def default_init_state(self):
        phase_vector = np.ndarray(shape=(self.dynamic_vars_N,3), dtype=float)
        # q1: alpha
        phase_vector[0][0] = 1.5
        # p1: alpha momentum
        phase_vector[0][1] = 4.8
        # f1: alpha generalized force (external wrt the system)
        phase_vector[0][2] = 0.0
        return phase_vector

    # provides an external forces to the system at current state
    # NOTE: usually indirectly issued by non-potential forces,
    #   like friction, thrust, etc. - all the stuff which didn't
    #   make it directly to the system hamiltonian.
    def current_ext_forces(self):
        return np.zeros(shape=(self.dynamic_vars_N, 1), dtype=float)

    # Same as current_ext_forces(...) but provides the external
    # forces for the given phase space vector @at_state
    def ext_forces(self, at_state):
        return np.zeros(shape=(self.dynamic_vars_N, 1), dtype=float)

    # @self.phase_vector a phase space vector
    #   which describes the system state
    #   NOTE: first index selects the generalized variable
    #         second index == 0 selects the generalized coordinate
    #                      == 1 selects the corresponding generalized momentum
    #                      == 2 selects the corresponding external force
    def __init__(self, params=None, init_phase_vector=None):
        if params is None:
            self.params = Pendulum.default_params()
        else:
            self.params = params

        if init_phase_vector is None:
            self.phase_vector = self.default_init_state()
        else:
            self.phase_vector = init_phase_vector

        self.H0 = self.H()
        print("Initialized Pendulum system: %s" % (str(self)))

    # RETURNS:
    #   Human readable representation
    def __str__(self):
        return "Phase vector: %s\nParameters: %s\nHamiltonian: %s\n" % (str(self.phase_vector), str(self.params), str(self.H()))

    # Hamiltonian of the system (dynamical external forces are ignored
    # here, cause random H shift which comes from current forces
    # being used in computation gives us just nothing, cause its
    # derivatives only matter)
    # NOTE: meanwhile the forces are to be computed for
    #   Hamiltonian derivatives.
    # @state target phase vector to compute the H in
    def H(self, state=None):
        if state is None:
            state = self.phase_vector
        m = self.params["m_kg"]
        g = self.params["g_mps2"]
        l = self.params["l_m"]
        q_alpha = state[0][0]
        p_alpha = state[0][1]
        return (m * g * l * (1.0 - cos(q_alpha))
                + m * (l * p_alpha)**2.0/2)

    # Partial derivative of our Hamiltonian
    # wrt to var with given index
    # @state target phase vector to compute the H in
    def dHdq(self, var_idx, state=None):
        if state is None:
            state = self.phase_vector
        if (var_idx == 0):
            m = self.params["m_kg"]
            g = self.params["g_mps2"]
            l = self.params["l_m"]
            q_alpha = state[0][0]
            f_alpha = state[0][2]

            #                                    Immediate
            #             Internal               Arbitrary
            #       ----Conservative-----      -External-
            #      /       force         \    /  force /
            return m * g * l * sin(q_alpha) + f_alpha
        raise RuntimeError("Unknown variable index: %s" % str(var_idx))

    # Partial derivative of our Hamiltonian
    # wrt to var momentum with given index
    # @state target phase vector to compute the H in
    def dHdp(self, var_idx, state=None):
        if state is None:
            state = self.phase_vector
        if (var_idx == 0):
            m = self.params["m_kg"]
            l = self.params["l_m"]
            p_alpha = state[0][1]
            return m * l**2 * p_alpha
        raise RuntimeError("Unknown variable index: %s" % str(var_idx))

    # Make a simulation step on @dt, while applying
    # given external generalized forces on the system.
    # TODO: take into account generalized forces
    def step(self, eneralized_forces_vector, dt):
        # initial energy at the beginning of the step
        H1 = self.H()

        # current control forces at the current step
        self.phase_vector[:,2] = self.current_ext_forces()

        # first to columns are dp and dq
        # last column is the work of external forces
        dV = np.ndarray(shape=self.phase_vector.shape, dtype=float)
        for i in range(self.phase_vector.shape[0]):
            # Hamilton equations for evolution
            # dq/dt = dH/dp
            # dp/dt = -dH/dq
            # ->
            # dq = (dH/dp) * dt
            # dp = -(dH/dq) * dt

            # first we jump directly to raw solution, which
            # is not optimized due to finite dt, and might have
            # energy discrepancy, we use it as 0th-approximation
            # to correct solution
            dp = -self.dHdq(i) * dt
            dq = self.dHdp(i) * dt

            # preliminary change of the phase vector
            dV[i, 0] = dq
            dV[i, 1] = dp

            # the work of the external forces
            dV[i, 2] = dq * self.phase_vector[i,2]

        # updating all q and p
        self.phase_vector[:,0] += dV[:,0]
        self.phase_vector[:,1] += dV[:,1]

        # computing energy gain/loss due to external forces
        # work along/against the system
        H_estimated = H1 + np.sum(dV[:,2])
        print("Ext. force work: %f" % (H_estimated - H1))

        # NOTE: when crossing the H-domains optimization
        #   of this kind strongly spoils the results
        #
        # for now fits only to the conservative systems
        #self.apply_H_correction(H_estimated)

    # Make a simulation step on @dt, but using balanced
    # scheme on integration at each integration step
    def step_v2(self, eneralized_forces_vector, dt):
        # initial energy at the beginning of the step
        H1 = self.H()

        # A the initial system locaion at the beginning of
        # the step.
        # B is the computed location at the end of the step.
        A = np.copy(self.phase_vector)
        B = np.copy(self.phase_vector)

        # current control/external forces at the current step
        fA = self.ext_forces(at_state=A)
        fB = self.ext_forces(at_state=B)

        # Hamilton equations for evolution
        # dq/dt = dH/dp
        # dp/dt = -dH/dq
        # ->
        # dq = (dH/dp) * dt
        # dp = -(dH/dq) * dt
        # Using the symmetrical integration iterative scheme:
        epsilon = 0.000001
        dB_max = 10 * epsilon
        iterations_count = 0
        while dB_max > epsilon:
            for i in range(self.phase_vector.shape[0]):
                B_prev = np.copy(B)
                fB = self.ext_forces(at_state=B_prev)
                # Updating q
                B[i,0] = A[i,0] + 0.5 * dt * (self.dHdp(i, state=A) + self.dHdp(i, state=B_prev))
                # Updating p
                B[i,1] = A[i,1] + 0.5 * dt * (-self.dHdq(i, state=A) - self.dHdq(i, state=B_prev)
                                              + fA[i] + fB[i])
                dB_max = np.max(np.abs(B - B_prev))
                iterations_count += 1

        print("Optimization iterations: %d" % iterations_count)

        # work along/against the system
        dA = np.dot((fA + fB)/2.0, (B-A)[:,0])
        print("Ext. force work: %f" % (dA))

        self.phase_vector = B


    # corrects the current phase space location to
    # fit H=H_estimated constraint using the gradient descent
    # with decay
    def apply_H_correction(self, H_estimated):
        print("Starting correction %f -> %f" % (self.H(), H_estimated))

        # how much of grad is determined with the current grad
        grad_decay = 0.1
        # how big step wrt to grad
        step_rate = 0.000001
        # The allowed error in H
        H_epsilon = 0.00000001

        def loss():
            return (self.H() - H_estimated)**2

        # @i the coordinate variable index
        def dLdq(i):
            return 2 * (self.H() - H_estimated) * self.dHdq(i)

        # @i the momentum variable index
        def dLdp(i):
            return 2 * (self.H() - H_estimated) * self.dHdp(i)
        
        # initializing the gradient vector with initial grad
        gradient = np.ndarray(shape=self.phase_vector.shape, dtype=float)
        for i in range(self.phase_vector.shape[0]):
            gradient[i,0] = self.dHdq(i)
            gradient[i,1] = self.dHdp(i)

        # iterating over down to min of loss function with decaying
        # gradient descent variation
        MAX_OPTIMIZATION_STEPS = 1000
        steps_counter = 0
        while loss() > H_epsilon:
            if (steps_counter > MAX_OPTIMIZATION_STEPS):
                print("WARNING: MAX_OPTIMIZATION_STEPS(%d) reached! Abort!"
                      % MAX_OPTIMIZATION_STEPS)
                break
            gradient = np.ndarray(shape=self.phase_vector.shape, dtype=float)
            for i in range(self.phase_vector.shape[0]):
                gradient[i,0] = dLdq(i) * grad_decay + gradient[i,0] * (1.0 - grad_decay)
                gradient[i,1] = dLdp(i) * grad_decay + gradient[i,1] * (1.0 - grad_decay)
            self.phase_vector = self.phase_vector - step_rate * gradient
            steps_counter += 1

        print("H correction done: %d steps, error: %f" % (steps_counter, loss()))
            
    # Runs some iterations
    def run(self, steps_count, dt=0.001):
        # only 2d plotting
        curve_p = np.ndarray(shape=(steps_count, 2), dtype=float)
        curve_q = np.ndarray(shape=(steps_count, 2), dtype=float)

        for i in range(steps_count):
            # V1 quite inaccurate
            #self.step(None, dt)
            # V2 waaay more precise
            self.step_v2(None, dt)
            print(self)

            # only 2d for now
            curve_q[i] = self.phase_vector[0][0]
            curve_p[i] = self.phase_vector[0][1]

        plt.plot(curve_q, curve_p)
        plt.show()



if __name__ == "__main__":
    argparse
    system = Pendulum()
    system.run(13000)
