using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2

# timestep minimum
#1/\kappa k^2

# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
function adv_closure(x)
  ℱ = plan_fft(x, 1)
  ℱ⁻¹ = plan_ifft(x, 1)
  function adv(c,um,κ,k)
    ch = ℱ * c
    tmp = -im*k.*ch.*um-κ*k.*k.*ch
    dc=real.(ℱ⁻¹ * tmp) #d/dx --> =-ik
    return dc
  end
end

function rk4(h, c, u,κ, k)
    k1 = adv2(c, u, κ, k)
    k2 = adv2(c + h/2*k1, u, κ, k)
    k3 = adv2(c + h/2*k2, u, κ, k)
    k4 = adv2(c + h*k3, u, κ, k)
    dc = h/6*(k1+2*k2+2*k3+k4)
    return dc
end

κ = 0.01     # "subgrid" kappa
dt = 4*2/(κ*1024^2)#1/5250

N=10
av=1/N

# setup grid
x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

adv2 = adv_closure(zeros(x_length, N))

ox=ones(x_length,1);
velocities = [1, 10, 100, 0.1, 1] #, 1, 10]
magnitudes = [0.9, 0.7, 0.5, 0.1]
lambdas = [0.01, 0.1, 1, 10, 100]
U_force = 1

for U_force in velocities
for mag in magnitudes
for λ in ProgressBar(lambdas)
  # initiate velocities
    u=randn(1,N) #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5

    # initial concentration
    c= ones(x_length,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
    dc = copy(c)
    # concentration bias
    Δconc = mag*cos.(x)
    Δconc = Δconc*ones(1, N);
    # plotting parameters

    t=0
    tmax=1000
    t_array = collect(t:dt:tmax);
    t_indices = round.(Int, collect(1:length(t_array)/tmax:length(t_array)))
    save_times = t_array[t_indices]

    #fig = Figure()
    #ax = Axis(fig[1, 1], xlabel = L"x",
    #  xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    #  xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
    #lines!(x,mean(c,dims=2)[:], label = L"\overline{c}")
    #lines!(x,(1/N)*c*u'[:], label = L"\overline{uc}") #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
    #lines!(x,Δconc[:, 1], label = L"\Delta(x)")
    #axislegend()#position = :rt, bgcolor = (:grey90, 0.25));
    #fig

    cs=zeros(x_length, tmax);
    fs=zeros(x_length, tmax);
    for t in ProgressBar(t_array)
      if minimum(abs.(save_times .- t)) == 0
        if t > 0
          #f2 = Figure()
          #ax = Axis(f2[1, 1])
          #lines!(ax, x,mean(c,dims=2)[:])
          #lines!(ax, x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
          #display(f2)
          cs[:, round(Int, t)]= mean(c,dims = 2)
          fs[:, round(Int, t)]= (1/N)*c*u'
        end
      end
      dc .= rk4(dt, c, U_force*u, κ, k)
      @. c = c +  dc + λ*dt*(c)*(1-c/(1 + Δconc))

      #@. c = c +  dc*dt + λ*dt*(c)*(1-c/(1 + Δconc))
      if any(isnan, c)
        break
      end

      𝒩 = randn(1,N)
      @. u = u - dt*u+sqrt(2)*𝒩
      ind = findall(u -> abs(u) >= 5, u)
      u[ind] .= 5
    end

    c_mean = mean(cs[:,201:end-1],dims = 2); #<c>
    flux_mean =mean(fs[:,201:end-1],dims = 2); #<uc>
    c_squared_mean = mean(cs[:,201:end-1].^2,dims = 2); #<c>
    gc=real(ifft(im*k[:,1].*fft(c_mean)));

    #c_mean = mean(cs,dims = 2); #<c>
    #flux_mean =mean(fs, dims = 2); #<uc>
    #c_squared_mean = mean(cs.^2,dims = 2); #<c>
    #gc=real(ifft(im*k[:,1].*fft(c_mean)));

    #f3 = Figure()
    #ax = Axis(f3[1, 1])
    #lines!(ax, c_squared_mean[:])  
    #display(f3) 

    
  #f3 = Figure()
  #ax = Axis(f3[1, 1])
  #lines!(ax, gc[:],flux_mean[:])
  #display(f3)

  
  save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * ".jld2"
  @save save_name c_mean flux_mean c_squared_mean gc
end

end

end
