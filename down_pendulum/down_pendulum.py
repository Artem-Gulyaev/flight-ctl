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

        print("Initialized Pendulum system: %s" % (str(self)))

    # RETURNS:
    #   Human readable representation
    def __str__(self):
        return "Phase vector: %s\nParameters: %s\nHamiltonian: %s\n" % (str(self.phase_vector), str(self.params), str(self.H()))

    # Hamiltonian of the system
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
            
    # Runs some iterations
    def run(self, steps_count, dt=0.01):
        for i in range(steps_count):
            self.step(None, dt)
            print(self)

if __name__ == "__main__":
    system = Pendulum()
    system.run(100)
