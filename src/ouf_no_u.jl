using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2

# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
function diff_closure(x)
  ℱ = plan_fft(x, 1)
  ℱ⁻¹ = plan_ifft(x, 1)
  function diff(c, κ, k, dt, Δ)
    ch = ℱ * c
    c0 = 1 .+ Δ
    @. ch = (ch/dt + λ*(1+ch)*(1-(1+ch)/c0))/(1/dt - κ*k^2)
    c_new = real.(ℱ⁻¹ * ch)
    return c_new
  end
end

κ=0.01    # "subgrid" kappa
λ=0.05    # relaxation to forcing
x_length = 1024
dt=2/(κ*x_length^2) # need 1/dt > κ(x_length/2)^2

N=10
av=1/N
uamp=0.1

x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

diff2 = diff_closure(zeros(x_length, N))

ox=ones(x_length,1);
magnitudes = [0.7] #, 0.1, 0.1]
divisor = [1] #, 3, 6, 13, 25]#,1, 3 63, 125]

div = 1
mag = 0.7
for mag in magnitudes
for div in divisor #ProgressBar(divisor)
    
    # initial concentration
    c=x*zeros(1,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)
    if any(isnan, c)
      print("nan error")
      break
    end
    # concentration bias
    Δconc = mag*cos.(div*x)
    Δconc = Δconc*ones(1, 10);
    
    # plotting parameters
    t=0
    tmax=1000
    #tmax = 10
    tpl=1 # plots every timestep  = 1
    t_array = collect(t:dt:tmax)

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
      if rem(t,tpl)==0
        if t > 0
          #f2 = Figure()
          #ax = Axis(f2[1, 1])
          #lines!(ax, x,mean(c,dims=2)[:])
          #lines!(ax, x,(1/N)*c*u'[:]) #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
          #title(num2str(t));
          #display(f2)
          cs[:, Int(t)]= mean(c,dims = 2)
          #print(c[100])
        end
      end
      c .= diff2(c, κ, k, dt, Δconc)
      
    end
  cf=mean(cs[:,201:end],dims = 2);
  
  save_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_kappa_" * string(κ)* "nou_FT.jld2"

  @save save_name cs cf
end

end