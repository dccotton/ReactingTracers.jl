using Statistics
using FFTW, GLMakie, ProgressBars
using ReactingTracers
using JLD2

# timestep minimum
#1/\kappa k^2
#1\k^2

function adv(c,um,kappa,k)
    ch=fft(c)#
    dc=real(ifft(-im*k.*ch.*um-kappa*k.*k.*ch)) #d/dx --> =-ik
    return dc
end

# function closure, a function that returns a function 
# this is a closure because it closes over the variables x, ℱ, ℱ⁻¹
function adv_closure(x)
  ℱ = plan_fft(x, 1)
  ℱ⁻¹ = plan_ifft(x, 1)
  function adv(c,um,kappa,k)
    ch = ℱ * c
    tmp = -im*k.*ch.*um-kappa*k.*k.*ch
    dc=real.(ℱ⁻¹ * tmp) #d/dx --> =-ik
    return dc
  end
end

varu=0.1   # variance of u
r=0.2    # damping rate in OE
kappa=0.01    # "subgrid" kappa
la=0.05    # relaxation to forcing
dt=1/5250
rfac=sqrt(2*varu*dt*r)

N=10
av=1/N
uamp=0.1

# slighly different choice of x to Glenn
x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

adv2 = adv_closure(zeros(x_length, N))

ox=ones(x_length,1);
#magnitudes = [0.7, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6]
magnitudes = [0.7, 0.5, 0.1]
divisor = [1, 3, 6, 13, 25]#, 63, 125]
for mag in magnitudes
for div in ProgressBar(divisor)
    # initiate velocities
    u=randn(1,N)*varu #returns a (N by 1) array of random numbers drawn from the standard normal distribution.
    ind = findall(u -> abs(u) >= 5, u)
    u[ind] .= 5

    # initial concentration
    c=x*zeros(1,N) #array of zeros, depth N, width x (i.e. conc at each point in x for each stochastic choice of v)

    # concentration bias
    #mag = 0.1;
    #Δconc = mag*sin.(2*pi/div*x)
    Δconc = mag*cos.(div*x)
    Δconc = Δconc*ones(1, 10);
    size(Δconc)
    # plotting parameters
    t=0
    tmax=1000
    tpl=1 # plots every timestep  = 1
    t_array = collect(t:dt:tmax)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"x",
      xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
      xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10, title = "")
    lines!(x,mean(c,dims=2)[:], label = L"\overline{c}")
    lines!(x,(1/N)*c*u'[:], label = L"\overline{uc}") #plots mean (c) and mean (uc) at each x value, u' = (1 x N), c = (N x length(x)) so matrix multiplication to give u'c = (1 x length(x))
    lines!(x,Δconc[:, 1], label = L"\Delta(x)")
    axislegend()#position = :rt, bgcolor = (:grey90, 0.25));
    fig

    cs=zeros(x_length, tmax);
    fs=zeros(x_length, tmax);
    c_p= copy(c); # c_p = c makes it so that c_p points to the same c
    dc = copy(c)

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
          fs[:, Int(t)]= (1/N)*c*u'
        end
      end
      #  tmp=c;
      @. c = 1.5*c-0.5*c_p;
      c_p .= c #get previous timestep, not sure why glenn has added this line?
      dc .= adv2(c,ox*u,kappa,k)
      #@. c = c +  dc*dt + la*dt*(-c-c .^2 + Δconc +c.*Δconc)./(1 .+ Δconc)
      @. c = c +  dc*dt + la*dt*(1 + c)*(1-abs(1+c)/(1 + Δconc))
      #@. c = c +  dc*dt + la*dt*(-c-c .^2 + Δconc +c.*Δconc)./(1 .+ Δconc)
      #c = c +  dc*dt + la*dt*(1+c).*(1-abs(1+c)./(1+delta));
      𝒩 = randn(1,N)
      @. u = u - r*dt*u+rfac*𝒩
      # ind = findall(u -> abs(u) >= 5, u)
      # u[ind] .= 5
    end
    #cf=mean(cs[:,2:end],dims = 2)
  #ff=mean(fs[:,2:end],dims = 2)
  cf=mean(cs[:,201:end],dims = 2);
  ff=mean(fs[:,201:end],dims = 2);
  gc=real(ifft(im*k[:,1].*fft(cf)));

  f3 = Figure()
  Axis(f3[1, 1])
  lines!(ax, gc[:],ff[:])
  #title(num2str(t));
  display(f3)

  save_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_FT.jld2"

  @save save_name cs fs ff cf gc
end

end