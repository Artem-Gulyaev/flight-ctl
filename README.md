# Just nothing but having fun with system dynamics control and prediction

# The overview.

In general within system modeling following entities can be
highlighted:

* The System (an active agent)
* External (w.r.t. the system) Environment

Further decomposition gives:

* The System can be **fuzzly** divided into

  * The Model
  * The Control System

  The division is very fuzzy, cause, basically, control system
  is just interactive part of the model,
  which just can talk human, but by other means gravity
  is the same control system just without any human input:
  it keeps the Moon at the relatively stable orbit around
  the Earth - good control system might do the same.

* Environment
  * Scalar loss function (can vary with time):
        Can formalize any arbitrary goal (like fuel
        economy, noisiness, travel time, etc.)

# The System

System is described by its current state and
routines to compute the next state at time t + dt.

These routines include the laws of Physics as well
as control system algorithms, resulting in the complex
behaviour of the System with the aim of minimization
loss function.

Note: the control system algorithms can have some
 parameters (hyper-parameters), which are externally
 defined, as well as other parameters which the system
 identifies by itself at the runtime. Hyper parameters
 can be fixed, and can be adjustible within the training
 sequence. Something, like internal routines and
 hardcoded parameters are fixed for the system.

System state vector: V(t)
System description: {evolution routines}


