# plotting script for microscopic and mesoscopic ring network models 
# script is called from: microscopic_model.jl, run_meanfield.jl
# December 2022 Bastian Pietras

using PyCall
@pyimport matplotlib.gridspec as gspec

fig = figure(figsize = (7, 6),constrained_layout="True")
gs = gspec.GridSpec(4,2,figure=fig)

element(i,j) = get(gs, (i,j))
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)


ax1 = fig.add_subplot(element(0,slice(0,2)))
ax2 = fig.add_subplot(element(1,slice(0,2)), sharex=ax1)
ax3 = fig.add_subplot(element(2,0))
ax4 = fig.add_subplot(element(2,1))
ax5 = fig.add_subplot(element(3,0))
ax6 = fig.add_subplot(element(3,1))


im1 = ax1.plot(time,Abar)
ax1.set_ylabel("Average\nactivity [Hz]")
# ax1.set_xlim([0,time[end]])
ax1.set_ylim([0,15])
setp(ax1[:get_xticklabels](),visible=false)

im22 = ax2.matshow(Avec', aspect="auto", origin="lower", extent=[0,time[end],-pi,pi],cmap="jet") #cmap="jet", #clim=[0,2.5],

ax2.set_ylabel("Place field\nposition [rad]")
fig.colorbar(im22, ax=ax2,location="right",pad=0.002,aspect=15)
ax2.set_xlim([40,50])
ax2.xaxis.set_ticks_position("bottom")

#panel C
ax3.hist(durations,bins=(0:0.006:0.6),align="left")
ax3.set_xlim([0,0.6])
ax3.set_xlabel("Event duration [s]")
ax3.set_ylabel("Count")

#panel D: linear regression for durations vs number of peaks
fpoly = Polynomials.fit(durations,n_peaks,1)
# println("slope n_peaks/duration= "*string(fpoly[1])*" vs 7.9 peaks/s (literature = 10.0)")

ax4.scatter(durations, n_peaks,s=5)
ax4.plot([0,0.6],[fpoly(0),fpoly(0.6)], color="r", label="linear regression")
ax4.set_ylim([0,5])
ax4.set_xlim([0,0.6])
ax4.set_xlabel("Event duration [s]")
ax4.set_ylabel("Number of peaks")

#panel E: linear regression for durations vs number of peaks
gpoly = Polynomials.fit(durations,abs.(distance),1)
# println("slope distance/duration= "*string(gpoly[1])*" vs 16.4 rad/s")

ax5.scatter(durations, abs.(distance),s=3)
ax5.plot([0,0.6],[gpoly(0),gpoly(0.6)], color="r", label="linear regression")
ax5.set_ylim([0,6.5])
ax5.set_xlim([0,0.6])
ax5.set_xlabel("Event duration [s]")
ax5.set_ylabel("Event path\nlength [rad]")

#panel F: event speed histogram
bump_speeds = abs.(distance[n_peaks .> 1]) ./ durations[n_peaks .> 1]
# println("average speed = "*string(mean(bump_speeds))*" vs 12.0 rad/s")

ax6.hist(bump_speeds,bins=(5:0.5:18),align="left")
ax6.set_xlabel("Event speed [rad/s]")
ax6.set_ylabel("Count")
ax6.set_xlim([5.1,18])

show()
