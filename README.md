# GME_lifecycle
Code and data for analysis and inference of general non-Markovian model of the mRNA life cycle, which is described 
in article "Stochastic modeling of the mRNA life process: A generalized master equation" .

## Directories
#### FSP_distribution.jl
Solving the generalized master equation using Finite State Projection (FSP) method.
#### delay_distribution.jl
Survival factor H(t) in the generalized master equation. 
#### noise.jl
Calculating the moments (including mean and variance) of mRNA number at steady state.
#### inference_function_GAL10.jl
Function for inferring transcriptional parameters from nascent mRNA distribution
#### generating_data.jl
Stochastic simulation for the non-Markovian model

