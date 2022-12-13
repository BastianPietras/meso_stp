# mesoscopic ring network models 
# script is called from: run_meanfield.jl
# December 2022 Bastian Pietras

using PyPlot, Random, RandomNumbers.Xorshifts, Distributions, FindPeaks1D, Polynomials, PyCall, DSP, NPZ

moving_average(data,n) = conv(data, ones(n))[1+round(Int,n/2):length(data)+round(Int,n/2)] ./ n

# Ninf = 1: Macroscopic model
# Ninf = 2: Diffusion model
# Ninf = 3: Jump-diffusion model (hybrid noise)
# Ninf = 4: 1st-order mean-field model [as in Schmutz, Gerstner, Schwalger (2020)]
function meanfield_model(Ninf,N,Tinit,Tend,tauR,J0,J1,U,mu)

    dt = 0.0001  #0.1 ms
    taum = 0.01  #10 ms
    L_ring = 2π  #length of ring
    M = 100      #number of populations on ring
    alpha = 1.0

    dx = L_ring/M
    Tfinal = Tinit + Tend
    time = 0 : dt : Tfinal
    dt_read = 0.005
    time_read = 0 : dt_read : Tfinal
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
    hvec = 0. .*rand(rng,M)
    xvec = 0. .*rand(rng,M)
    Qvec = 0. .*rand(rng,M)

    #######
    # functions
    #######
    f(I) = alpha .* log.(1 .+ exp.( I ./ alpha ) )
    dhdt(h, Isyn, mu, U, tau) = (1/tau)*(-h .+ mu) .+ U .* Isyn

    Avec = f.(hvec)
    h_readout[1,:] = hvec
    A_readout[1,:] = Avec
    ind_readout = 1

    if Ninf == 1 # macro = no noise
        println("macroscopic model")
        dxdt_macro(x,A,U,taur) = (1 .- x) ./ taur .- U .* x .* A

        for t = 2 : length(time)

            t_now = time[t]

            Isyn = wJmat * (xvec .* Avec)
            hvec += dt * dhdt(hvec,Isyn,mu,U,taum)
            xvec += dt * dxdt_macro(xvec,Avec,U,tauR)
            Avec = f.(hvec)
            Amean_vec = vcat(mean(Avec),Amean_vec[1:navec-1])

            if t == ind_readout*dt_factor
                h_readout[ind_readout+1,:] = hvec
                A_readout[ind_readout+1,:] = Avec
                Amean_readout[ind_readout+1] = mean(Amean_vec)
                ind_readout += 1
            end
        end

    elseif Ninf == 2 # Valentin's meso-model
        println("Diffusion model")
        dxdt_val(x,Qsyn,U,taur) = (1 .- x) ./ taur .- U .* Qsyn
        dQdt_val(Q,x,A,U,taur) = 2 .* (x .- Q) ./ taur .- U .* (2 .- U) .* Q .* A

        fh = zeros(M)
        sfn = zeros(M)
        qsyn = zeros(M)

        for t = 2 : length(time)

            t_now = time[t]

            fh = f.(hvec)
            sfn = sqrt.(fh ./ N ./ dt) .* randn(rng,M)
            qsyn = xvec .* fh .+ sqrt.( max.(Qvec , 0)) .* sfn
            Isyn = wJmat * qsyn
            hvec += dt * dhdt(hvec,Isyn,mu,U,taum)
            xvec += dt * dxdt_val(xvec,qsyn,U,tauR)
            Qvec += dt * dQdt_val(Qvec,xvec,fh,U,tauR)
            # Avec = fh .+ sfn
            Avec = rand.(rng, Binomial.(N, 1 .- exp.(-fh .* dt ))) ./ N ./ dt
            Amean_vec = vcat(mean(Avec),Amean_vec[1:navec-1])

            if t == ind_readout*dt_factor
                h_readout[ind_readout+1,:] = hvec
                A_readout[ind_readout+1,:] = Avec
                Amean_readout[ind_readout+1] = mean(Amean_vec)
                ind_readout += 1
            end

        end
        
    elseif Ninf == 3 # \tildeQ-model
        println("Hybrid model")
        dxdt_tilde(x,Qsyn,U,taur) = (1 .- x) ./ taur .- U .* Qsyn
        dQdt_tilde(Q,x,fh,U,taur) = -(2 ./ taur .+ U .* (2 .- U) .* fh) .* Q .+ (U .* x) .^ 2 .* fh
   
        fh = zeros(M)
        sfn = zeros(M)
        qsyn = zeros(M)
            
        for t = 2 : length(time)

            t_now = time[t]
            fh = f.(hvec)
            Avec = rand.(rng, Binomial.(N, 1 .- exp.(-fh .* dt ))) ./ N ./ dt
            sfn = sqrt.(fh ./ N ./ dt) .* randn(rng,M)
            qsyn = xvec .* Avec .+ sqrt.(max.(Qvec,0)) .* sfn 
            Isyn = wJmat * qsyn
            hvec += dt * dhdt(hvec,Isyn,mu,U,taum)
            xvec += dt * dxdt_tilde(xvec,qsyn,U,tauR)
            Qvec += dt * dQdt_tilde(Qvec,xvec,fh,U,tauR)
            Amean_vec = vcat(mean(Avec),Amean_vec[1:navec-1])

            if t == ind_readout*dt_factor
                h_readout[ind_readout+1,:] = hvec
                A_readout[ind_readout+1,:] = Avec
                Amean_readout[ind_readout+1] = mean(Amean_vec)
                ind_readout += 1
            end

        end
        
    elseif Ninf == 4 # first-order meso-model
        println("1st-order meanfield-model")
        dxdt_val1st(x,A,U,taur) = (1 .- x) ./ taur .- U .* x .* A

        fh = zeros(M)
        sfn = zeros(M)

        for t = 2 : length(time)

            t_now = time[t]

            fh = f.(hvec)
            sfn = sqrt.(fh ./ N ./ dt) .* randn(rng,M)
            Avec = fh .+ sfn
            # Avec = rand.(rng, Binomial.(N, max.(1 .- exp.(-fh .* dt ),0)))
            Isyn = wJmat * (xvec .* Avec)
            hvec += dt * dhdt(hvec,Isyn,mu,U,taum)
            xvec += dt * dxdt_val1st(xvec,Avec,U,tauR)
            Amean_vec = vcat(mean(Avec),Amean_vec[1:navec-1])

            if t == ind_readout*dt_factor
                h_readout[ind_readout+1,:] = hvec
                A_readout[ind_readout+1,:] = Avec
                Amean_readout[ind_readout+1] = mean(Amean_vec)
                ind_readout += 1
            end

        end
    end

    time_out = time_read[time_read .>= Tinit] .- Tinit
    hvec_out = h_readout[time_read .>= Tinit,:]
    Avec_out = A_readout[time_read .>= Tinit,:]
    Abar_out = Amean_readout[time_read .>= Tinit,:]

    time_out, hvec_out, Avec_out, Abar_out, f

end

