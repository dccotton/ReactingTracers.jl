using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2
##
# timestep minimum
#1/\kappa k^2
##
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

function rk4(h, c, u, u_t0ph_2, u_t0ph, κ, k, U_force)
    k1 = adv2(c, U_force*u, κ, k)
    k2 = adv2(c + h/2*k1, U_force*u_t0ph_2, κ, k)
    k3 = adv2(c + h/2*k2, U_force*u_t0ph_2, κ, k)
    k4 = adv2(c + h*k3, U_force*u_t0ph, κ, k)
    dc = h/6*(k1+2*k2+2*k3+k4)
    return dc
end

κ = 0.001     # "subgrid" kappa
dt = 8*2/(8*κ*1024^2)#1/5250
#dt = 4*2/(8*κ*1024^2)#1/5250 for £kappa = 0.001

N=250
av=1/N

# setup grid
x_length = 512
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

adv2 = adv_closure(zeros(x_length, N))

ox=ones(x_length,1);
velocities = [1.0] #, 10, 100, 0.1, 1] #, 1, 10]
magnitudes = [0.7] #[0.9, 0.7, 0.5, 0.1]
lambdas = [1.0] #, 0.1, 0.01, 10.0]
#lambdas = [1, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, ] #[0.01, 0.1, 1, 10, 100, 0.5, 1.5, 2]
#lambdas = [2, 3, 5, 7, 1.7, 1.4, 1.8, 1.6, 1.9, 2.5, 3.5, 4, 8, 9]
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
    tmax= 1000
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
    c_total = zeros(x_length, N, tmax)
    cs=zeros(x_length, tmax);
    fs=zeros(x_length, tmax);
    umat = zeros(N, tmax);
    for t in ProgressBar(t_array)
      if minimum(abs.(save_times .- t)) == 0
        if t > 0
          #f2 = Figure()
          #ax = Axis(f2[1, 1])
          #lines!(ax, x,mean(c,dims=2)[:])
          #lines!(ax, x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
          #display(f2)
          c_total[:, :, round(Int, t)] = c
          cs[:, round(Int, t)]= mean(c,dims = 2)
          fs[:, round(Int, t)]= (1/N)*c*u'
          umat[:, round(Int, t)]=u
        end
      end
      #dc .= rk4(dt, c, U_force*u, κ, k)
      u_t0ph_2 = u .- dt/2*u .+ sqrt(dt)*randn(1,N)
      u_t0ph = u_t0ph_2 .- dt/2*u_t0ph_2 .+ sqrt(dt)*randn(1,N)
      dc .= rk4(dt, c, u, u_t0ph_2, u_t0ph, κ, k, U_force)
      #print(maximum(dc))
      #print("u_max = ", maximum(u_t0ph_2))
      @. c = c +  dc + λ*dt*(c)*(1-c/(1 + Δconc))
      u = u_t0ph

      #@. c = c +  dc*dt + λ*dt*(c)*(1-c/(1 + Δconc))
      if any(isnan, c)
        print("nan")
        break
      end

      #𝒩 = randn(1,N)
      #@. u = u - dt*u+sqrt(2)*𝒩
      ind = findall(u -> abs(u) >= 5, u)
      u[ind] .= 5
    end

    c_mean = mean(cs[:,101:end-1],dims = 2); #<c>
    flux_mean =mean(fs[:,101:end-1],dims = 2); #<uc>
    c_squared_mean = mean(cs[:,101:end-1].^2,dims = 2); #<c^2>
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

  
  save_name = "mag_" * string(mag) * "_U_" * string(U_force) * "_lambda_" * string(λ) * "_k_" * string(κ) * ".jld2"
  @save save_name c_mean flux_mean c_squared_mean gc cs fs umat c_total
end

end

end