# simulation of microscopic ring network model
# of M populations equally distributed along a ring of length L_ring
# each consisting of N homogeneous LNP neurons with short-term plasticity
# December 2022 Bastian Pietras

# this script calls analyze_data.jl if analyze_on == 1, otherwise it plots only the timeseries of the population activities
analyze_on = 0  


using PyPlot, Random, RandomNumbers.Xorshifts, Distributions, FindPeaks1D, Polynomials, PyCall, DSP, NPZ


mu = -1.4 # noise-induced
# mu=-0.9 # fatigue-induced

# Tend = 1950 #in seconds for fatigue-induced NLEs 
# Tend = 4000 #in seconds for noise-induced NLEs with N = 50
Tend = 20
L_ring = 2π


M = 100 # number of grid points / units
# M = 50 # to speed up sims w/o changing collective dynamics much
N = 50 # number of neurons per population, unless Ninf == 1

tauD = 0.8   # depression time constant (0.8s)
# tauD = 0.33 # for more than one revolution
J0 = 1300.0  # uniform feedback inhibition
J1 = 3000.0  # map-specific interaction
U = 0.8

Tinit = 5.0
T_final = Tinit + Tend

function spikingnetField(Tend,L_ring,tauR,J0,J1,U,mu)

    dt = 0.0001  #0.1 ms
    taum = 0.01  #10 ms
    alpha = 1.0

    dx = L_ring/M
    time = 0 : dt : Tend
    dt_read = 0.005
    time_read = 0 : dt_read : Tend
    dt_factor = round(Int,dt_read/dt)

    h_readout = zeros((length(time_read),M))
    A_readout = zeros((length(time_read),M))
    Amean_readout = zeros(length(time_read))
    navec=100
    Amean_vec = zeros(navec)

    #######
    # coupling matrix
    #######
    wJij = (i,j) -> (-J0 .+ J1 .* cos.((i-j)*dx))
    wJmat = zeros((M,M))
    for i = 1:M
        for j = 1:M
            wJmat[i,j] = wJij(i,j) / M
        end
    end
    wJnorm = sum(wJmat[1,:])

    # rng = Xoroshiro128Star(1234)
    rng = Xoroshiro128Star(rand(Int))

    #######
    # initial conditions
    #######
    hvec = 0. .*randn(rng,M)
    xvec = 0. .*randn(rng,(M,N))
    xs = copy(xvec)

    #######
    # functions
    #######
    f(I) = alpha .* log.(1 .+ exp.( I ./ alpha ) )
    dhdt(h, Isyn, mu, U, tau) = (1/tau)*(-h .+ mu) .+ U .* Isyn
    dhdt_micro(h, Isyn, mu, U, tau) = (1/tau)*(-h .+ mu)
    dxdt(x,xs,U,taur) = (1 .- x) ./ taur .- U .* xs
    dxdt_micro(x,xs,U,taur) = (1 .- x) ./ taur

    Avec = f.(hvec)
    h_readout[1,:] = hvec
    A_readout[1,:] = Avec
    ind_readout = 1

    n_active = zeros(M)

    for t = 2 : length(time)

        t_now = time[t]

        # active units per population
        n_active = rand.(rng,Binomial.( N , 1 .- exp.(-f.(hvec) .* dt) ))
        Avec = n_active ./ N ./ dt

        for i = 1 : M

            neurons_active = zeros(N)
            neurons_active[sample(rng,1:N,n_active[i],replace=false, ordered=false)] .= 1
            xs[i,:] = xvec[i,:] .* neurons_active
            xvec[i,:] += dt * dxdt_micro.(xvec[i,:],xs[i,:],U,tauR) .- U .* xs[i,:]

        end

        Isyn = wJmat * mean(xs,dims=2)
        hvec += dt * dhdt_micro(hvec,Isyn,mu,U,taum) .+ U .* Isyn

        Amean_vec = vcat(mean(Avec),Amean_vec[1:navec-1])

        if t == ind_readout*dt_factor
            h_readout[ind_readout+1,:] = hvec
            A_readout[ind_readout+1,:] = Avec
            Amean_readout[ind_readout+1] = mean(Amean_vec)
            ind_readout += 1
        end

    end

    time_read, h_readout, A_readout, Amean_readout, f

end


# quick run to compile code
spikingnetField(0.01, L_ring,tauD,J0,J1,U,mu)
println(".")

# run & benchmark network with specified parameters
GC.gc()

println("tauD = "*string(tauD))
println("J0 = "*string(J0))
println("J1 = "*string(J1))
println("U = "*string(U))
println("mu = "*string(mu))

time_vec, hvec_all, Avec_all, Abar_all, func = @time spikingnetField(T_final, L_ring,tauD,J0,J1,U,mu)

time = time_vec[time_vec .>= Tinit] .- Tinit
hvec = hvec_all[time_vec .>= Tinit,:]
Avec = Avec_all[time_vec .>= Tinit,:]
Abar = Abar_all[time_vec .>= Tinit]

moving_average(data,n) = conv(data, ones(n))[1+round(Int,n/2):length(data)+round(Int,n/2)] ./ n

if analyze_on == 1
    name = "micro_M"*string(M)*"_N"*string(N)*"_T"*string(Tend)*"_mu"*string(mu)*"_tauD"*string(tauD)
    npzwrite("data_"*name*".npz",Dict("time" => time,"hvec" => hvec, "Abar" => Abar))
    
    include("analyze_data.jl") # need time, hvec as inputs
else
    h, axs = subplots(2,1,figsize=(8,4),constrained_layout="True", sharex="all",dpi=200)

    im1 = axs[1].plot(time,Abar)
    axs[1].set_xlim([0,time[end]])
    axs[1].set_ylim([0,18])

    im2 = axs[2].matshow(func.(hvec)', aspect="auto", origin="lower", extent=[0,time[end],-3.15,3.15],cmap="jet",vmin=0,vmax=50) #cmap="jet", #clim=[0,2.5],
    #             interpolation = "bessel", extent=[0,time[end],0,N_neurons])
    axs[2].xaxis.set_ticks_position("bottom")
    h.colorbar(im2, ax=axs[2],location="right",aspect=20, pad=.005)
    # tight_layout()

    axs[1].set_title(L"Microscopic model $-$ depression $\tau_D=$"*string(tauD),pad=10)
    axs[1].set_ylabel("Average activity [Hz]")
    axs[2].set_ylabel("Place field position")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_yticks([-π,0,π],["-π","0","π"])

    show()
end

