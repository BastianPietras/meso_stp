# simulation of meanfield ring network models
# of M=100 populations equally distributed along a ring of length 2π
# each consisting of N homogeneous LNP neurons with short-term plasticity
# December 2022 Bastian Pietras

# this script calls meanfield_models.jl
# if case == 2, the script also calls analyze_data.jl and plot_figure.jl
# otherwise it plots only the timeseries of the population activities

##########################
### run meanfield model on
### ring of length 2π with
### M=100 populations ####
##########################
### choose model #########
Ninf = 3 # 1: macro, 2: diffusion model, 3: jump-diffusion (hybrid), 4: 1st MF 
##########################
### neurons per population
N = 50 # if Ninf == 1, then N → ∞ automatically
##########################
### choose option ########
case = 1 # run short time series and plot outcome
# case = 2 # run long time series, analyze and plot outcome
##########################
Tinit = 10.0
Tend = 200.0 #in seconds
# Tend = 4000.0 #in seconds
# Tend = 2350.0 #in seconds
# Tend = 1950.0 #in seconds
##########################
### parameters ###########
J0 = 1300.0  # uniform feedback inhibition
J1 = 3000.0  # map-specific interaction

### noise-induced regime 
tauD = 0.8  # in s
U = 0.8     # depletion variable
mu = -1.4   # external constant input

# ### fatigue-induced regime 
# tauD = 0.8  # in s
# U = 0.8     # depletion variable
# mu = -0.9   # external constant input

include("meanfield_models.jl")

# run & benchmark network with specified parameters
# quick run to compile code
meanfield_model(Ninf,10,0.1,0.1,tauD,J0,J1,U,mu)
println(".")
GC.gc()

if Ninf == 1
    println("Macroscopic simulation, N_pop → ∞")
elseif Ninf == 2 || Ninf == 3 || Ninf == 4
    println("Mesoscopic simulation, N_pop = "*string(N))
end
println("tauD = "*string(tauD))
println("J0 = "*string(J0))
println("J1 = "*string(J1))
println("U = "*string(U))
println("mu = "*string(mu))

time, hvec, Avec_ts, Abar_ts, func = @time meanfield_model(Ninf,N,Tinit,Tend,tauD,J0,J1,U,mu)

if case == 1
    h, axs = subplots(2,1,figsize=(10,4),constrained_layout="True", sharex="all",dpi=200)
    im1 = axs[1].plot(time,Abar_ts)
    axs[1].set_xlim([0,time[end]])
    axs[1].set_ylim([0,18])

    im3 = axs[2].matshow(func.(hvec)', aspect="auto", origin="lower", extent=[0,time[end],-3.15,3.15],cmap="jet",vmin=0,vmax=50) #cmap="jet", #clim=[0,2.5],
    axs[2].xaxis.set_ticks_position("bottom")
    h.colorbar(im3, ax=axs[2],location="right",aspect=20, pad=.005)
    axs[1].set_title(L"Input $\mu=$"*string(mu)*L", Depression $\tau_D=$"*string(tauD),pad=10)
    axs[1].set_ylabel("Average activity [Hz]")
    axs[2].set_ylabel("Place field position")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_yticks([-π,0,π],["-π","0","π"])
    show()
elseif case == 2
    include("analyze_data.jl")      # we need [time, hvec] as inputs for this function
    include("plot_figure.jl")
end
