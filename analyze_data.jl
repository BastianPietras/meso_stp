# data analysis script for microscopic and mesoscopic ring network models 
# inputs: time (vector of size 1xL) and population-specific input potentical hvec (vector of size MxL); typically, M=100
# December 2022 Bastian Pietras

using PyPlot, Random, RandomNumbers.Xorshifts, Distributions, FindPeaks1D, Polynomials, PyCall, DSP, NPZ, StatsBase, PyCall

moving_average(data,n) = conv(data, ones(n))[1+round(Int,n/2):length(data)+round(Int,n/2)] ./ n


function analyze_data(time,hvec)
    
    M=100
    L_ring = 2pi

    alpha=1.0
    f(I) = alpha .* log.(1 .+ exp.( I ./ alpha ) )

    Avec = f.(hvec)
    Amean = mean(Avec,dims=2)
    Abar = moving_average(Amean,5)

    b_thres = mean(Abar)
    # b_thres_stop = 0.3
    # b_thres_stop = 1.65
    b_thres_stop = b_thres

    pkindices, properties = findpeaks1d(Abar,height=1.0,prominence=nothing,width=0.1,relheight=0.8);

    L = length(time)
    maxAloc = zeros(L)
    maxA = zeros(L)
    maxAloc_binned = zeros(L)
    maxA_binned = zeros(L)
    Abinned = zeros(size(Avec))

    for j in 1:M
        Abinned[:,j] = moving_average(Avec[:,j],20)
    end

    cvec = zeros(ComplexF64,L)
    cvecb = zeros(ComplexF64,L)
    for j in 1:L
        cvec[j] = mean(moving_average(Avec[j,:],1) .* exp.(1im .* L_ring/M*(1:M)))
        cvecb[j] = mean(Abinned[j,:] .* exp.(1im .* L_ring/M*(1:M)))
    end


    for k in 1:L
        local Aloc_coord, Aloc_cbinned
        Aloc_coord = argmax(Avec[k,:])
        Aloc_cbinned = argmax(Abinned[k,:])
        maxA[k] = Avec[k,Aloc_coord]
        maxAloc[k] = Aloc_coord * L_ring/M - L_ring/2

        maxA_binned[k] = Avec[k,Aloc_cbinned]
        maxAloc_binned[k] = Aloc_cbinned * L_ring/M - L_ring/2
    end


    bursts = Abar .> b_thres
    diff_bursts = diff(bursts)
    start_ind = findall(diff_bursts .> 0)
    stop_ind = findall(diff_bursts .< 0) 

    if stop_ind[1] < start_ind[1]
        stop_ind = stop_ind[2:end]
    end

    if stop_ind[end] < start_ind[end]
        start_ind = start_ind[1:end-1]
    end

    start_vals = zeros(Int,0)
    stop_vals = zeros(Int,0)

    append!(start_vals,start_ind[1])
    append!(stop_vals,stop_ind[1])

    for start in start_ind
        local stop
        # start = start_ind[k]
        if start > stop_vals[end]
            append!(start_vals,convert(Int,start))
            try 
                stop = findfirst(Abar[start+1:end] .< b_thres_stop) + start
            catch
                continue
            end
            # append!(stop_vals,stop - findfirst(Abar[stop:-1:1] .> b_thres) )
            stop = stop - findfirst(Abar[stop:-1:1] .> b_thres) + 1
            append!(stop_vals,convert(Int,stop))
        end
    end

    n_bursts = length(start_vals)


    n_peaks = zeros(n_bursts)
    durations = zeros(n_bursts)
    distance = zeros(n_bursts)
    Alocation = mod.(maxAloc,2pi)
    Aloc_binned = mod.(maxAloc_binned,2pi)

    event_direction = zeros(0)


    for k in 1:n_bursts
        local a_trajec, start, stop, ind_burst

        start = start_vals[k]    
        stop = stop_vals[k]
        # stop = findfirst(Abar[start+1:end] < b_thres_stop) + start
        # stop = stop - findfirst(Abar[stop:-1:1] > b_thres) 

        ind_burst = start:stop
        n_peaks[k] = count(i->(start <= i <= stop), pkindices)
        durations[k] = time[stop] - time[start]

        a_trajec = mod.(mod.(diff(Alocation[ind_burst]).+2pi,2pi).+pi,2pi).-pi
        distance[k] = sum(a_trajec)

        # analyze NLEs
        if n_peaks[k] > 1      
            append!(event_direction, sign(distance[k]))
        end
    end

    # determine interburst intervals
    IBI = zeros(n_bursts-1)
    for k in 1:n_bursts-1
        IBI[k] = time[start_vals[k+1]] - time[stop_vals[k]]
    end

    println("total bursts = "*string(n_bursts))
    fpoly = Polynomials.fit(durations,n_peaks,1)
    println("slope n_peaks/duration= "*string(fpoly[1])*" vs 7.9 peaks/s (literature = 10.0)")
    gpoly = Polynomials.fit(durations,abs.(distance),1)
    println("slope distance/duration= "*string(gpoly[1])*" vs 16.4 rad/s")

    
    T1 = mean(IBI)
    T2 = mean(IBI.^2) 
    T3 = mean(IBI.^3)
    T4 = mean(IBI.^4)
    CV = std(IBI)/T1
    println("mean(IBI) = " *string(T1))
    println("CV(IBI) = "*string(CV))
    # cumulants
    k1 = T1
    k2 = T2 - k1^2
    k3 = T3 - 3*k1*k2 - k1^3
    k4 = T4 - 4*k1*k3-3*k2^2-6*k1^2*k2-k1^4

    γs = k3*k2^(-3/2)
    αs = γs/(3*CV)
    γe = k4*k2^(-2)
    αe = γe/(15*CV^2)
    println("check CV with cumulants = "*string(CV-sqrt(k2)/k1))
    println("skewness(IBI) = "*string(γs))
    println("resc. skewness(IBI) = "*string(αs))
    println("kurtosis(IBI) = "*string(γe))
    println("resc. kurtosis(IBI) = "*string(αe))
    println("")

    bursts_long_ind = findall(n_peaks .> 1)
    n_bursts_long = length(bursts_long_ind)
    SWfraction = round(n_bursts_long/n_bursts,digits=3)
    fwfraction = round(sum(event_direction .> 0) / n_bursts_long,digits=3)

    println("# long bursts = "*string(n_bursts_long))
    println("fraction of SW-events = "*string(SWfraction))
    println("fraction of forward-events = "*string(fwfraction))    
    
    abs_speed = abs.(distance[n_peaks .> 1]) ./ durations[n_peaks .> 1]
    println("average abs speed = "*string(mean(abs_speed))*" vs 12.0 rad/s")

    println("Serial correlations (bw/fw) = "*string(autocor(event_direction,1:5)))

    speeds = (distance[n_peaks .> 1]) ./ durations[n_peaks .> 1]
    println("Serial correlations (speed) = "*string(autocor(speeds,1:5)))



    # estimated step size analysis

    ess_0 = zeros(0)
    ess_1 = zeros(0)
    ess_2 = zeros(0)
    ess_predicted = zeros(0)

    phi_now = zeros(1)
    for k in bursts_long_ind
        global phi_now

        local start, stop, ind_burst, a0, a1, a2
        local m1, m2, burst_thres

        start = start_vals[k]    
        stop = stop_vals[k]
        # stop = findfirst(Abar[start+1:end] < b_thres_stop) + start
        # stop = stop - findfirst(Abar[stop:-1:1] > b_thres)     
        ind_burst = start:stop

        m1 = mean(Abar[ind_burst])
        m2 = mean(Avec[:])
        burst_thres = (m1+3m2)/4

        a0 = zeros(0)
        a1 = zeros(0)
        a2 = zeros(0)
        for tt in ind_burst     
            append!(a0, angle(cvec[tt]))
            append!(a1, angle(cvecb[tt]))

            if Abar[tt] <= burst_thres && mean(diff(Abar[tt-3:tt])) < 0 
                append!(a2, phi_now)
            else
                phi_now = angle(cvecb[tt])
                append!(a2, phi_now)
            end    
        end

        append!(ess_0, diff(unwrap(a0)))
        append!(ess_1, diff(unwrap(a1)))
        append!(ess_2, diff(unwrap(a2)))

        av_step_length = (distance[k] / length(ind_burst))
        append!(ess_predicted, av_step_length .* ones(length(ind_burst)))
    end

    return time[time .<= 300], Avec[time .<= 300,:], Abar[time .<= 300], ess_2, ess_predicted, IBI, event_direction, durations, n_peaks, distance

end

# data = npzread("data_micro_fatigue-induced_N50_T2350_J0_1300_J1_3000.npz")
# println(".")

# time = data["time"]
# hvec = data["hvec"]

time,Avec,Abar,event_step_size,event_step_size_predicted,IBI,event_direction,durations,n_peaks,distance = analyze_data(time,hvec);
