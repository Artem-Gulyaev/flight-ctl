#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import *

class Pendulum:

    # RETURNS: default parameters for the system
    def default_params():
        return {
                "g_mps2": 9.81
                , "l_m": 1.0
                , "m_kg": 1.0
               }

    # RETURNS: default initial state for the system
    def default_init_state():
        phase_vector = np.ndarray(shape=(1,2), dtype=float)
        # alpha
        phase_vector[0][0] = 1.0
        # alpha momentum
        phase_vector[0][1] = 0.0
        return phase_vector

    # @self.phase_vector a phase space vector
    #   which describes the system state
    #   NOTE: first index selects the generalized variable
    #         second index == 0 selects the generalized coordinate
    #                      == 1 selects the corresponding moment
    def __init__(self, params=None, init_phase_vector=None):
        if params is None:
            self.params = Pendulum.default_params()
        else:
            self.params = params

        if init_phase_vector is None:
            self.phase_vector = Pendulum.default_init_state()
        else:
            self.phase_vector = init_phase_vector

        self.H0 = self.H()
        print("Initialized Pendulum system: %s" % (str(self)))

    # RETURNS:
    #   Human readable representation
    def __str__(self):
        return "Phase vector: %s\nParameters: %s\nHamiltonian: %s\n" % (str(self.phase_vector), str(self.params), str(self.H()))

    # Hamiltonian of the system
    # TODO: take it into account at computing to adjust
    #    phase vector to keep H constant (surely in case of
    #    no external forces)
    def H(self):
        m = self.params["m_kg"]
        g = self.params["g_mps2"]
        l = self.params["l_m"]
        q_alpha = self.phase_vector[0][0]
        p_alpha = self.phase_vector[0][1]
        return (m * g * l * (1.0 - cos(q_alpha))
                + m * (l * p_alpha)**2.0/2)

    # Partial derivative of our Hamiltonian
    # wrt to var with given index
    def dHdq(self, var_idx):
        if (var_idx == 0):
            m = self.params["m_kg"]
            g = self.params["g_mps2"]
            l = self.params["l_m"]
            q_alpha = self.phase_vector[0][0]
            return m * g * l * sin(q_alpha)
        raise RuntimeError("Unknown variable index: %s" % str(var_idx))

    # Partial derivative of our Hamiltonian
    # wrt to var momentum with given index
    def dHdp(self, var_idx):
        if (var_idx == 0):
            m = self.params["m_kg"]
            l = self.params["l_m"]
            p_alpha = self.phase_vector[0][1]
            return m * l**2 * p_alpha
        raise RuntimeError("Unknown variable index: %s" % str(var_idx))

    # Make a simulation step on @dt, while applying
    # given external generalized forces on the system.
    # TODO: take into account generalized forces
    def step(self, eneralized_forces_vector, dt):
        for i in range(self.phase_vector.shape[0]):
            # Hamilton equations for evolution
            # dq/dt = dH/dp
            # dp/dt = -dH/dq
            # ->
            # dq = (dH/dp) * dt
            # dp = -(dH/dq) * dt

            dp = - self.dHdq(i) * dt
            dq = self.dHdp(i) * dt

            self.phase_vector[i, 0] += dq
            self.phase_vector[i, 1] += dp

        # for now fits only to the conservative systems
        self.apply_H_correction()

    # corrects the current phase space location to
    # fit H=const constraint
    def apply_H_correction(self):
        # how much of grad is determined with the current grad
        grad_decay = 0.1
        # how big step wrt to grad
        step_rate = 0.0001
        # The allowed error in H
        H_epsilon = 0.00001

        def loss():
            return (self.H() - self.H0)**2

        # @i the coordinate variable index
        def dLdq(i):
            return 2 * (self.H() - self.H0) * self.dHdq(i)

        # @i the momentum variable index
        def dLdp(i):
            return 2 * (self.H() - self.H0) * self.dHdp(i)
        
        # initializing the gradient vector with initial grad
        gradient = np.ndarray(shape=self.phase_vector.shape, dtype=float)
        for i in range(self.phase_vector.shape[0]):
            gradient[i,0] = self.dHdq(i)
            gradient[i,1] = self.dHdp(i)

        # iterating over down to min of loss function
        steps_counter = 0
        while loss() > H_epsilon:
            gradient = np.ndarray(shape=self.phase_vector.shape, dtype=float)
            for i in range(self.phase_vector.shape[0]):
                gradient[i,0] = dLdq(i) * grad_decay + gradient[i,0] * (1.0 - grad_decay)
                gradient[i,1] = dLdp(i) * grad_decay + gradient[i,1] * (1.0 - grad_decay)
            self.phase_vector = self.phase_vector - step_rate * gradient
            steps_counter += 1

        print("H correction done: %d steps, error: %f" % (steps_counter, loss()))
            
    # Runs some iterations
    def run(self, steps_count, dt=0.01):
        for i in range(steps_count):
            self.step(None, dt)
            print(self)

if __name__ == "__main__":
    system = Pendulum()
    system.run(100)
