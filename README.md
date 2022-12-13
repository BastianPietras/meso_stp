# meso_stp
Code for our manuscript "Mesoscopic description of hippocampal replay and metastability in spiking neural networks with short-term plasticity"


To run the Julia codes, open your terminal and type:<br />
a) for simulating the microscopic model: "julia microscopic_model.jl"<br />
b) for simulating one of the mean-field models: "julia run_meanfield.jl"<br />


In the file "run_meanfield.jl", you have the option to choose:<br />
	1: the macroscopic model (when the network size N goes to infinity)<br />
	2: the diffusion model<br />
	3: the jump-diffusion model (with hybrid noise)<br />
	4: a first-order mean-field model (as in Schmutz, Gerstner and Schwalger, J. Math. Neurosci. 2020)<br />


To run the codes correctly, make sure that you have all the files in the same folder.
