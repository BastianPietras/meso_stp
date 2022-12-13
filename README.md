# meso_stp
Code for our manuscript "Mesoscopic description of hippocampal replay and metastability in spiking neural networks with short-term plasticity"

To run the Julia codes, open your terminal and type:
a) for simulating the microscopic model: "julia microscopic_model.jl"
b) for simulating one of the mean-field models: "julia run_meanfield.jl"

In the file "run_meanfield.jl", you have the option between:
1: the macroscopic model (when the network size N goes to infinity)
2: the diffusion model
3: the jump-diffusion (with hybrid noise)
4: a first-order mean-field model (as in Schmutz, Gerstner and Schwalger, J. Math. Neurosci. 2020)

To run the codes correctly, make sure that you have all the files in the same folder.
