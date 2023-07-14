using DataFrames, GLM
using GLMakie
using PerceptualColourMaps

# find max|<c'>|
# find K_eff
# find <c'^2>/<c>^2

# NOTE WITH u = 0, cs is <c> = 1 + c' whist with u != 0, cs = c'

stat_type = 3 # 1: find range|<c'>|, 2: find K_eff, 3: find <c'^2>/<c>^2, 4: find <c>
plot_type = 2 # 1: plot with mag on x, 2: plot with k_forcing on x, 3: plot with κ on x
kappa_variable = 2 # 1: plot with mag, 2: plot with k_forcing, 3: plot with both
no_u = false #true

λ = 0.05 
varu=0.1   # variance of u
r=0.2    # damping rate in OE
markers = ["circle", "rect", "diamond", "hexagon", "cross", "xcross", "utriangle", "dtriangle"];

if no_u
    magnitudes = [0.01, 0.1, 0.7]
    kappas = sort([10, 1, 0.5, 0.25, 0.15, 0.05, 0.01, 0.075, 0.025, 0.005, 0.001])
else
    magnitudes = [0.01, 0.05, 0.1, 0.5, 0.7]
    kappas = [0.01]
end
k_forcing = [1, 3, 6, 13, 25]
k_choice_indx = 5
mag_choice_indx = 1
#κ = 0.01

if kappa_variable == 1
    var_name = "_k_" * string(k_forcing[k_choice_indx])
elseif kappa_variable == 2
    var_name = "_mag_" * string(magnitudes[mag_choice_indx])
else
    var_name = "_both"
end

function axis_labels(stat_type)
    if stat_type == 1
        y_label = "range⟨c'⟩"
    elseif stat_type == 2
        y_label = L"\Kappa_{eff}"
    elseif stat_type == 3
        y_label = "max⟨c'^2⟩/⟨c⟩^2"
    else
        y_label = "<c>"
    end
    return y_label
end

function obtain_stat(stat_type, cs, cf, ff = [], gc = [])
    if stat_type == 1
        stat = maximum(cf) - minimum(cf)
    elseif stat_type == 2
        model = lm(@formula(y ~ x), DataFrame(x=gc[:], y=ff[:]))
        coeffs = coef(model)  
        stat = -coeffs[2]
    elseif stat_type == 3
        stat = maximum(mean(cs[:,201:end].^2,dims = 2)[:]./(1 .+ cf[:]).^2)
    else
        if ff == []
            stat = mean(cf[:])
        else
            stat = mean(cf[:])+1
        end
    end
    return stat
end

function axis_title(stat_type, plot_type, κ, no_u, choice = "")
    if stat_type == 1
        stat_type_lab = "range_c_prime_"
    elseif stat_type == 2
        stat_type_lab = "k_eff_"
    elseif stat_type == 3
        stat_type_lab = "c_prime_square_over_c_squared_"
    else
        stat_type_lab = "mean_c_prime_"
    end
    if plot_type == 1 || kappa_variable == 1
        plot_lab = "mag_"
    elseif plot_type == 2 || kappa_variable == 2
        plot_lab = "k_"
    else
        plot_lab = "κ_"
    end
    if no_u
        u_lab = "_no_u" * choice
    else
        u_lab = string(κ)
    end
    title = stat_type_lab * plot_lab * u_lab * ".png"
    return title
end

# load in the data
matrix = zeros(length(kappas), length(magnitudes), length(k_forcing));
for kappa_indx in 1:length(kappas)
    for m_indx = 1:length(magnitudes)
        mag = magnitudes[m_indx]
        for n_indx = 1:length(k_forcing)
            div = k_forcing[n_indx]
            if no_u
                load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(kappas[kappa_indx]) * "_nou_FT.jld2"
                @load load_name cs cf
                var = obtain_stat(stat_type, cs, cf)
            else
                load_name = "mag_" * string(mag) * "_k_" * string(round(div; sigdigits =  3)) * "_kappa_" * string(kappas[kappa_indx]) * "_FT.jld2"
                @load load_name cs fs ff cf gc
                var = obtain_stat(stat_type, cs, cf, ff, gc)                
            end
            
            matrix[kappa_indx ,m_indx, n_indx] = var #m_indx + n_indx #var
        end
    end
end

fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]
if plot_type == 3
    if stat_type == 1
        ax = Axis(fig[1, 1], xscale = log10, yscale = log10)
    else
        ax = Axis(fig[1, 1], xscale = log10)
    end
else
    ax = Axis(fig[1, 1], yscale = log10)
end

# plot the data
if plot_type == 1
    for indx = 1:length(k_forcing)
        colours = specify_colours(length(k_forcing))
        lines!(magnitudes, matrix[1, :, indx], label = "L = " * string(round(2*pi/k_forcing[indx]*r/varu; sigdigits =  3)) * "⟨v⟩/r", color = colours[indx])
    end
    ax.xlabel = "|Δ|"
elseif plot_type == 2
    for indx = 1:length(magnitudes)
        colours = specify_colours(length(magnitudes))
        lines!(2*pi./k_forcing*r/varu, matrix[1, indx, :], label = "|Δ| = " * string(magnitudes[indx]), color = colours[indx])
    end
    ax.xlabel = "L/(⟨u^2⟩/r)"
else
    if kappa_variable == 1
        for indx = 1:length(magnitudes)
            colours = specify_colours(length(magnitudes))
            lines!(kappas, matrix[:, indx, k_choice_indx], label = "|Δ| = " * string(magnitudes[indx]), color = colours[indx]) 
            scatter!(kappas, matrix[:, indx, k_choice_indx], markersize = 10, marker = :circle, color = colours[indx]) 
        end      
    elseif kappa_variable ==2
        for indx = 1:length(k_forcing)
            colours = specify_colours(length(k_forcing))
            lines!(kappas, matrix[:, mag_choice_indx, indx], label = "L = " * string(round(2*pi/k_forcing[indx]*r/varu; sigdigits =  3)) * "⟨v⟩/r", color = colours[indx])
            scatter!(kappas, matrix[:, mag_choice_indx, indx], markersize = 10, marker = :circle, color = colours[indx])
        end
    else
        for mag_indx = 1:length(magnitudes)
            colours = specify_colours(length(magnitudes))
            for k_indx = 1:length(k_forcing)
                lines!(kappas, matrix[:, mag_indx, k_indx], label = "L = " * string(round(2*pi/k_forcing[k_indx]*r/varu; sigdigits =  3)) * "⟨v⟩/r, |Δ| = " * string(magnitudes[mag_indx]), color = colours[mag_indx])
                scatter!(kappas, matrix[:, mag_indx, k_indx], color = "black") #, markersize = 10, marker = "circle") #markers[k_indx], markercolor = colours[mag_indx])
            end
        end
    end
    ax.xlabel = "κ"
end
ax.ylabel = axis_labels(stat_type)
if no_u
    if kappa_variable == 1
        ax.title = "λ = " * string(λ) * ", k = " * string(k_forcing[k_choice_indx])
    else
        ax.title = "λ = " * string(λ) * ", |Δ| = " * string(magnitudes[mag_choice_indx])
    end
else
    ax.title = "λ = " * string(λ) * ", r = " * string(r) * ", ⟨v^2⟩ = " * string(varu) * ", κ = " * string(κ)
end
axislegend()
save(axis_title(stat_type, plot_type, κ, no_u, var_name), fig)
