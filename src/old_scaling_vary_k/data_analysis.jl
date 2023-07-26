using Statistics
using DataFrames, GLM
using JLD2
using FileIO
using GLMakie
using PerceptualColourMaps
using ReactingTracers
using FFTW
GLMakie.activate!(inline=false)

# find max|<c'>|
# find K_eff
# find <c'^2>/<c>^2

# NOTE WITH u = 0, cs is <c> = 1 + c' whist with u != 0, cs = c'

stat_type = 3 # 1: find range|<c'>|, 2: find K_eff, 3: find <c'^2>/<c>^2, 4: find <c>, 5: find mean(<c>) - √(1-|Δ|^2), 6: find error d<uc>/dx - λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))
plot_type = 3 # 1: plot with mag on x, 2: plot with k_forcing on x, 3: plot with κ\λ on x
kappa_variable = 1 # 1: plot with mag, 2: plot with k_forcing, 3: plot with both mag and k, 4: plot with κ/λ # code doesn't work with 3 yet...

#λ = 0.05 
r = 0.2    # damping rate in OE
varu = r^2   # variance of u
markers = ['o', 'x', '+', '*', 's', 'd'] #, "hexagon", "cross", "xcross", "utriangle", "dtriangle"];

# setup grid
x_length = 1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

if varu == 0
    magnitudes = [0.01, 0.1, 0.7]
    kappas = sort([1, 10, 0.1, 0.01, 0.001, 0.025, 0.05, 0.15, 0.25, 0.5])
    #sort([10, 1, 0.5, 0.25, 0.15, 0.05, 0.01, 0.075, 0.025, 0.005, 0.001])
elseif varu == r^2
    magnitudes = [0.7] #[0.1, 0.5, 0.7, 0.9]
    lambdas = [0.1, 0.5, 1, 1.5, 5, 10] # [0.1, 0.5, 1, 5, 10]
    kappas = lambdas
else
    magnitudes = [0.01, 0.05, 0.1, 0.5, 0.7]
    magnitudes = [0.01, 0.1, 0.7]
    kappas = [0.01, 0.02]
end
#k_forcing = [1, 3, 25]
k_forcing = [1, 3, 6, 13, 25]
k_choice_indx = 2
mag_choice_indx = 3
lambda_or_kappa_choice_indx = 3

if kappa_variable == 1
    var_name = "_k_" * string(k_forcing[k_choice_indx])
elseif kappa_variable == 2
    var_name = "_mag_" * string(magnitudes[mag_choice_indx])
elseif kappa_variable == 3
    var_name = "_both"
else
    if varu == 0
        var_name = "_κ_" * string(kappes[lambda_or_kappa_choice_indx])
    else
        var_name = "_λ_" * string(lambdas[lambda_or_kappa_choice_indx])
    end
end

function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

function axis_labels(stat_type)
    if stat_type == 1
        y_label = "range⟨c'⟩"
    elseif stat_type == 2
        y_label = L"\Kappa_{eff}"
    elseif stat_type == 3
        y_label = "max⟨c'^2⟩/⟨c⟩^2"
    elseif stat_type == 4
        y_label = "<c>"
    elseif stat_type == 5
        y_label = "(mean(<c>) - √(1-|Δ|^2))/√(1-|Δ|^2)"
    else
        y_label = "||(d<uc>/dx - λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))||/||λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))||"
    end
    return y_label
end

function obtain_stat(stat_type, cs, cf, ff = [], gc = [])
    if cs[10, 1000] == 0
        cs = cs[:, 1:999]
    end
    if stat_type == 1
        if ff == []
            stat = maximum(cs[:, end]) - minimum(cs[:, end]) # want steady state
        else
            stat = maximum(cf) - minimum(cf)
        end   
    elseif stat_type == 2
        model = lm(@formula(y ~ x), DataFrame(x=gc[:], y=ff[:]))
        coeffs = coef(model)  
        stat = -coeffs[2]
    elseif stat_type == 3
        stat = maximum(mean(cs[:,201:end].^2,dims = 2)[:]./(1 .+ cf[:]).^2)
    else
        if ff == []
            #stat = mean(cf[:])
            stat = mean(cs[:, end])
        else
            stat = mean(cf[:])+1
        end
    end
    return stat
end

function obtain_stat_means(stat_type, c_mean, flux_mean, c_squared_mean, gc, mag, λ, div, k)
    if stat_type == 1
        stat = maximum(c_mean) - minimum(c_mean)
    elseif stat_type == 2
        model = lm(@formula(y ~ x), DataFrame(x=gc[:], y=flux_mean[:]))
        coeffs = coef(model)  
        stat = -coeffs[2]
    elseif stat_type == 3
        stat = maximum(abs.(c_mean.^2 - c_squared_mean)./c_mean.^2)
    elseif stat_type ==4
        stat = mean(c_mean[:])
    elseif stat_type == 5 # compare mean(<c>) to √(1-|Δ|^2)
        stat = (mean(c_mean[:]) - sqrt(1-mag^2))/sqrt(1-mag^2)
    elseif stat_type == 6 # compare rms error of d<uc>/dx and λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))
        ∇uc = real(ifft(im*k[:,1].*fft(flux_mean)))[:];
        λc = (λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(div*x))))[:]
        stat = sqrt(mean((∇uc .- λc).^2))/sqrt(mean(λc.^2))
    end
    return stat
end

function axis_title(stat_type, plot_type)
    if stat_type == 1
        stat_type_lab = "range_c_prime_"
    elseif stat_type == 2
        stat_type_lab = "k_eff_"
    elseif stat_type == 3
        stat_type_lab = "c_prime_square_over_c_squared_"
    elseif stat_type == 4
        stat_type_lab = "mean_c_prime_"
    elseif stat_type == 5
        stat_type_lab = "mean_c_to_delta_"
    else
        stat_type_lab = "flux_to_guess_"
    end
    if kappa_variable !=3
        if plot_type !=1 && kappa_variable !=1
            fixed_var = "|Δ| = " * string(magnitudes[mag_choice_indx])
        elseif plot_type !=2 && kappa_variable !=2
            fixed_var = "k = " * string(k_forcing[k_choice_indx])
        else
            if varu == 0
                fixed_var = "κ = " * string(kappas[lambda_or_kappa_choice_indx])
            else
                fixed_var = "λ = " * string(kappas[lambda_or_kappa_choice_indx])
            end
        end
    else
        fixed_var = ""
    end

    if plot_type == 1
        plot_lab = "mag_on_x_"
    elseif plot_type == 2 || kappa_variable == 2
        plot_lab = "k_on_x_"
    else
        if varu == 0
            plot_lab = "κ_on_x_"
        else
            plot_lab = "λ_on_x_"
        end
    end

    title = stat_type_lab * plot_lab * fixed_var * ".png"
    return title
end

# load in the data
matrix = zeros(length(kappas), length(magnitudes), length(k_forcing));
for kappa_indx in 1:length(kappas)
    for m_indx = 1:length(magnitudes)
        mag = magnitudes[m_indx]
        for n_indx = 1:length(k_forcing)
            div = k_forcing[n_indx]
            if varu == r^2
                data_folder = "data/u_rat_1"
                data_name = "mag_" * string(mag) * "_k_" * string(round(div, sigdigits = 3)) * "_lambda_" * string(kappas[kappa_indx]) * "_FT.jld2"
                # Concatenate the folder and file name to get the full path
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name c_mean flux_mean c_squared_mean gc
                    var = obtain_stat_means(stat_type, c_mean, flux_mean, c_squared_mean, gc, mag, kappas[kappa_indx], div, k)
                catch systemerror
                    print("no file named " * data_name)
                    var = 0
                end
                
            elseif varu == 0
                #load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(kappas[kappa_indx]) * "_nou_FT.jld2"
                data_folder = "data/u_0"
                data_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(kappas[kappa_indx]) * "_lambda_" * string(λ) * "_nou_FT.jld2"
                load_name = joinpath(data_folder, data_name)
                try
                    @load load_name cs cf
                    var = obtain_stat(stat_type, cs, cf)
                catch systemerror
                    print("no file named" * data_name)
                    var = 0
                end                    
            else
                load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(kappas[kappa_indx]) * "_FT.jld2"
                @load load_name cs fs ff cf gc
                var = obtain_stat(stat_type, cs, cf, ff, gc)                
            end
            
            matrix[kappa_indx ,m_indx, n_indx] = var #m_indx + n_indx #var
        end
    end
end
matrix = replace(matrix) do x
    x == 0 ? NaN : x
end

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]
if plot_type == 3 && varu == 0
    if stat_type == 1
        ax = Axis(fig[1, 1], xscale = log10, yscale = log10)
    else
        ax = Axis(fig[1, 1], xscale = log10)
    end
elseif stat_type ==3
    ax = Axis(fig[1, 1], yscale = log10)
else
    ax = Axis(fig[1, 1]) #, yscale = log10)
    #ax = Axis(fig[1, 1], yscale = log10)
end

# sizes
legendsize = 30;
axlabelsize = 30;
axtitlesize = 40;

# plot the data
if plot_type == 1
    if kappa_variable == 2
        for indx = 1:length(k_forcing)
            colours = specify_colours(length(k_forcing))
            scatterlines!(magnitudes, matrix[lambda_or_kappa_choice_indx, :, indx], label = "L = " * string(round(2*pi/k_forcing[indx]*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r", color = colours[indx], )
        end
    elseif kappa_variable == 4
        for indx = 1:length(kappas)
            colours = specify_colours(length(kappas))
            if varu == 0
                scatterlines!(magnitudes, matrix[indx, :, k_choice_indx], label = "κ = " * string(kappas[indx]), color = colours[indx],)
            else
                scatterlines!(magnitudes, matrix[indx, :, k_choice_indx], label = "λ = " * string(kappas[indx]), color = colours[indx],)
            end
        end        
    ax.xlabel = "|Δ|"
    end
elseif plot_type == 2
    if kappa_variable == 1
        for indx = 1:length(magnitudes)
            colours = specify_colours(length(magnitudes))
            scatterlines!(2*pi./k_forcing*r/sqrt(varu), matrix[lambda_or_kappa_choice_indx, indx, :], label = "|Δ| = " * string(magnitudes[indx]), color = colours[indx])
        end
    elseif kappa_variable == 4
        for indx = 1:length(kappas)
            colours = specify_colours(length(kappas))
            if varu == 0
                scatterlines!(2*pi./k_forcing*r/sqrt(varu), matrix[indx, mag_choice_indx, :], label = "κ = " * string(kappas[indx]), color = colours[indx],)
            else
                scatterlines!(2*pi./k_forcing*r/sqrt(varu), matrix[indx, mag_choice_indx, :], label = "λ = " * string(kappas[indx]), color = colours[indx],)
            end
        end    
    end
    ax.xlabel = "L/(√⟨u^2⟩/r)"
else
    if kappa_variable == 1
        for indx = 1:length(magnitudes)
            colours = specify_colours(2) #length(magnitudes))
            scatterlines!(kappas, matrix[:, indx, k_choice_indx], label = "|Δ| = " * string(magnitudes[indx]), color = colours[indx]) 
        end      
    elseif kappa_variable ==2
        for indx = 1:length(k_forcing)
            colours = specify_colours(length(k_forcing))
            scatterlines!(kappas, matrix[:, mag_choice_indx, indx], label = "L = " * string(round(2*pi/k_forcing[indx]*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r", color = colours[indx])
        end
    else
        for mag_indx = 1:length(magnitudes)
            colours = specify_colours(length(k_forcing))
            for k_indx = 1:length(k_forcing)
                scatterlines!(kappas, matrix[:, mag_indx, k_indx], label = "L = " * string(round(2*pi/k_forcing[k_indx]*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r, |Δ| = " * string(magnitudes[mag_indx]), color = colours[k_indx], marker = markers[mag_indx])
                #scatter!(kappas, matrix[:, mag_indx, k_indx], color = "black", marker = markers[k_indx]) #, markersize = 10, marker = "circle") #markers[k_indx], markercolor = colours[mag_indx])
            end
        end
    end
    if varu == 0
        ax.xlabel = "κ"
    else
        ax.xlabel = "λ"
    end
end

if kappa_variable !=3
    if plot_type !=1 && kappa_variable !=1
        ax.title = "|Δ| = " * string(magnitudes[mag_choice_indx])
    elseif plot_type !=2 && kappa_variable !=2
        ax.title = "L = " * string(round(2*pi/k_forcing[k_choice_indx]*r/sqrt(varu); sigdigits =  3)) * "⟨v⟩/r"
    else
        if varu == 0
            ax.title = "κ = " * string(kappas[lambda_or_kappa_choice_indx])
        else
            ax.title = "λ = " * string(kappas[lambda_or_kappa_choice_indx])
        end
    end
end

ax.titlesize = axtitlesize
ax.ylabel = axis_labels(stat_type)
ax.xlabelsize = axlabelsize
ax.ylabelsize = axlabelsize

axislegend(labelsize = legendsize)

save(axis_title(stat_type, plot_type), fig)
