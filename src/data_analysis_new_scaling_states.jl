using Statistics
using DataFrames, GLM
using JLD2
using FileIO
using GLMakie
using PerceptualColourMaps
using ReactingTracers
using FFTW
using LsqFit
GLMakie.activate!(inline=false)

# find max|<c'>|
# find K_eff
# find <c'^2>/<c>^2

# NOTE WITH u = 0, cs is <c> = 1 + c' whist with u != 0, cs = c'

# Stat type lists
# 1: find range|<c>|
#2: find K_eff
#3: find ⟨<c'^2>(x)/(<c>^2(x)⟩
#4: find <c>
#5: find mean(<c>) - √(1-|Δ|^2)
#6: find error d<uc>/dx - λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))
#8: find max(⟨c⟩)
#9: find min(⟨c⟩)
#10: find x position of max(c)
#11: find the size of the sin(kx) and cos(kx) components for ⟨c'^2⟩
#12: find the size of the sin(kx) and cos(kx) components for ⟨c⟩
#13: find the size of the sin(kx) and cos(kx) components for ⟨c⟩ and ⟨c'^2⟩
#14: find the size of the sin(kx) and cos(kx) components for d⟨uc⟩/dx
#15: scaled ⟨c̅⟩
#16: scaled range c̅
#17: ⟨<c'^2>(x)/(<c>^2(x)⟩ scaled to 1
#18: scaled fourier coefficients for ⟨c'^2⟩

stat_type = 4
plot_type = 3 # 1: plot with mag on x, 2: plot with u on x, 3: plot with κ\λ on x
line_variable = 2 # 1: plot with mag, 2: plot with u, 3: plot with both mag and u, 4: plot with κ/λ, 5: plot with κ(with var u non u) # code doesn't work with 3 yet...
num_states_to_add = 2 # 0: no new states, 1: add 2 state, 2: add 2 and 3 state, etc.

states_to_add = [2, 3, 4, 10]#, 15, 20, 100]

kmax = 2 # in plotting the FT, the max number of sin and cosine fourier modes
multiple_data = false #true
data_line_colours = 1 # 1: colour the lines by the multiple data variable, 2: colour the lines by the line_variable

if stat_type == 11 || stat_type == 12 || stat_type == 14 || stat_type == 18
    print("here")
    global multiple_data
    multiple_data = true # allows multiple lines to be plotted
    multiple_data_number = 2*kmax + 1
    data_labels = fill("", 5)
    for indx = 1:multiple_data_number - 1
        if indx % 2 == 0
            data_labels[indx] = ", sin" * string(Int(indx/2)) * "x"
        else
            data_labels[indx] = ", cos" * string(Int(indx/2 + 0.5)) * "x"
        end
        data_labels[end] = ", 0"
    end
end

if stat_type == 13
    global multiple_data
    multiple_data == true # allows multiple lines to be plotted
    multiple_data_number = 2*(2*kmax + 1)
    data_labels = fill("", multiple_data_number)
    for indx = 1:Int(multiple_data_number/2 - 1)
        if indx % 2 == 0
            data_labels[indx] = "⟨c⟩ sin" * string(Int(indx/2)) * "x, "
            data_labels[indx + Int(multiple_data_number/2)] = "⟨c'^2⟩ sin" * string(Int(indx/2)) * "x, "
        else
            data_labels[indx] = "⟨c⟩ cos" * string(Int(indx/2 + 0.5)) * "x, "
            data_labels[indx + Int(multiple_data_number/2)] = "⟨c'^2⟩ cos" * string(Int(indx/2 + 0.5)) * "x, "
        end
        data_labels[end] = "⟨c'^2⟩ 0, "
        data_labels[Int(multiple_data_number/2)] = "⟨c⟩ 0, "
    end
end

markers = ['o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd', 'o', 'x', '+', '*', 's', 'd'] #, "hexagon", "cross", "xcross", "utriangle", "dtriangle"];

# setup grid
x_length = 512 #1024
x = nodes(x_length, a = -pi, b = pi)
k  = wavenumbers(x_length)

magnitudes = [0.1, 0.5, 0.7, 0.9]
#magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

lambdas = sort([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 100.0])
lambdas = sort([1.0, 1.5, 0.5, 0.1, 10, 0.01, 100, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 7.0])
#lambdas = [0.1, 1, 10]
kappas = [0.01, 0.001, 0.0001]

velocities = [1.0]#, 10, 100]
varu = 1
κ = 0.001

mag_choice_indx = 3
lambda_or_kappa_choice_indx = 3
vel_choice_indx = 1

# sizes
legendsize = 60;
axlabelsize = 40;
axtitlesize = 50;
axticksize = 40;

function specify_colours(num_colours)
    colours = cmap("Gouldian", N = num_colours)
    return colours
end

function axis_labels(stat_type)
    if stat_type == 1
        y_label = "range⟨c⟩"
    elseif stat_type == 2
        y_label = L"\Kappa_{eff}"
    elseif stat_type == 3
        y_label = "⟨⟨c'^2⟩(x)/c̅(x)^2⟩"
    elseif stat_type == 4
        y_label = "<c>"
    elseif stat_type == 5
        y_label = "(mean(<c>) - √(1-|Δ|^2))/√(1-|Δ|^2)"
    elseif stat_type == 6
        y_label = "||(d<uc>/dx - λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))||/||λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))||"
    elseif stat_type == 8
        y_label = "max⟨c⟩"
    elseif stat_type == 9
        y_label = "min⟨c⟩"
    elseif stat_type == 10
        y_label = "x pos max c"
    elseif stat_type == 11 || stat_type == 12 || stat_type == 13 || stat_type == 14
        y_label = "FT coeff size"
    elseif stat_type == 15
        y_label = "scaled ⟨c̅⟩"
    elseif stat_type == 16
        y_label = "scaled range(c̅)"
    else
        y_label = "normalised ⟨⟨c'^2⟩(x)/c̅(x)^2⟩"
    end
    return y_label
end

function obtain_stat_means(stat_type, c_mean, flux_mean, c_squared_mean, gc, mag, λ, k)
    if stat_type == 1
        stat = maximum(c_mean) - minimum(c_mean)
    elseif stat_type == 2
        model = lm(@formula(y ~ x), DataFrame(x=gc[:], y=flux_mean[:]))
        coeffs = coef(model)  
        stat = -coeffs[2]
    elseif stat_type == 3
        stat = mean(abs.(c_squared_mean .- c_mean.^2)./c_mean.^2)
    elseif stat_type ==4
        stat = mean(c_mean[:])
    elseif stat_type == 5 # compare mean(<c>) to √(1-|Δ|^2)
        stat = (mean(c_mean[:]) - sqrt(1-mag^2))/sqrt(1-mag^2)
    elseif stat_type == 6 # compare rms error of d<uc>/dx and λ⟨c⟩(1-⟨c⟩/(1+Δ(x)))
        ∇uc = real(ifft(im*k[:,1].*fft(flux_mean)))[:];
        λc = (λ*c_mean.*(1 .- c_mean./(1 .+ mag*cos.(x))))[:]
        stat = sqrt(mean((∇uc .- λc).^2))/sqrt(mean(λc.^2))
    elseif stat_type == 8
        stat = maximum(c_mean)
    elseif stat_type == 9
        stat = minimum(c_mean)
    elseif stat_type == 10
        stat = x[argmax(c_mean)]
    elseif stat_type == 11 || stat_type == 12 || stat_type == 14
        if stat_type == 11
            ft_cf=(fft(c_squared_mean .- c_mean.^2))[:]
        elseif stat_type == 12
            ft_cf=(fft(c_mean))[:]
        else
            ft_cf = (im*k[:,1].*fft(flux_mean))[:];
        end
        fourier_coeffs = zeros(2*kmax+1)
        for indx = 1:kmax
            if indx % 2 == 0
                fourier_coeffs[2*indx - 1] = real(ft_cf[indx+1])*2/x_length #cosine component
                fourier_coeffs[2*indx] = -imag(ft_cf[indx+1])*2/x_length #sin component
            else
                fourier_coeffs[2*indx-1] = -real(ft_cf[indx+1])*2/x_length
                fourier_coeffs[2*indx] = imag(ft_cf[indx+1])*2/x_length
            end   
        end
        fourier_coeffs[end] = real(ft_cf[1])/x_length
        stat = fourier_coeffs
    elseif stat_type == 13
        ft_cf_2=(fft(c_squared_mean .- c_mean.^2))[:]
        ft_cf=(fft(c_mean))[:]
        fourier_coeffs = zeros(2*(2*kmax+1))
        for indx = 1:kmax
            if indx % 2 == 0
                fourier_coeffs[2*indx - 1] = real(ft_cf[indx+1])*2/x_length #cosine component
                fourier_coeffs[2*indx] = -imag(ft_cf[indx+1])*2/x_length #sin component
                fourier_coeffs[2*indx - 1 + (2*kmax+1)] = real(ft_cf_2[indx+1])*2/x_length #cosine component
                fourier_coeffs[2*indx+ (2*kmax+1)] = -imag(ft_cf_2[indx+1])*2/x_length #sin component
            else
                fourier_coeffs[2*indx-1] = -real(ft_cf[indx+1])*2/x_length
                fourier_coeffs[2*indx] = imag(ft_cf[indx+1])*2/x_length
                fourier_coeffs[2*indx-1 + (2*kmax+1)] = -real(ft_cf_2[indx+1])*2/x_length
                fourier_coeffs[2*indx + (2*kmax+1)] = imag(ft_cf_2[indx+1])*2/x_length
            end   
        end
        fourier_coeffs[end] = real(ft_cf_2[1])/x_length
        fourier_coeffs[2*kmax+1] = real(ft_cf[1])/x_length
        stat = fourier_coeffs
    elseif stat_type ==15
        stat = (mean(c_mean[:]) - (1-mag^2)^0.5)/(1 - (1-mag^2)^0.5)
    elseif stat_type ==16
        stat = (maximum(c_mean) - minimum(c_mean))/(2*mag)
    elseif stat_type == 17
        stat = mean(abs.(c_squared_mean .- c_mean.^2)./c_mean.^2)
    end
    return stat
end

function axis_title(stat_type, plot_type, states)
    if stat_type == 1
        stat_type_lab = "range_c_"
    elseif stat_type == 2
        stat_type_lab = "k_eff_"
    elseif stat_type == 3
        stat_type_lab = "c_prime_square_over_c_squared_"
    elseif stat_type == 4
        stat_type_lab = "mean_c_prime_"
    elseif stat_type == 5
        stat_type_lab = "mean_c_to_delta_"
    elseif stat_type == 6
        stat_type_lab = "flux_to_guess_"
    elseif stat_type == 8
        stat_type_lab = "max_c_"
    elseif stat_type == 9
        stat_type_lab = "min_c_"
    elseif stat_type == 10
        stat_type_lab = "max_c_pos_"
    elseif stat_type == 11
        stat_type_lab = "⟨c'2⟩ fourier coefficients_"
    elseif stat_type == 12
        stat_type_lab = "⟨c⟩ fourier coefficients_"
    elseif stat_type == 13
        stat_type_lab = "⟨c⟩ and ⟨c'^2˔ fourier coefficients_"
    elseif stat_type == 14
        stat_type_lab = "d⟨uc⟩_dx fourier coefficients_"
    elseif stat_type == 15
        stat_type_lab = "scaled ⟨c̅⟩_"
    elseif stat_type == 16
        stat_type_lab = "scaled range c̅_"
    else
        stat_type_lab = "normalised_c_prime_square_over_c_squared_"
    end
    if line_variable !=3
        if plot_type !=1 && line_variable !=1
            fixed_var = "|Δ| = " * string(magnitudes[mag_choice_indx])
        elseif plot_type !=2 && line_variable !=2
            fixed_var = "U = " * string(velocities[vel_choice_indx])
        else
            if varu == 0
                fixed_var = "κ = " * string(lambdas[lambda_or_kappa_choice_indx])
            else
                fixed_var = "λ = " * string(lambdas[lambda_or_kappa_choice_indx])
            end
        end
    else
        fixed_var = ""
    end

    if plot_type == 1
        plot_lab = "mag_on_x_"
    elseif plot_type == 2 || line_variable == 2
        plot_lab = "U_on_x_"
    else
        if varu == 0
            plot_lab = "κ_on_x_"
        else
            plot_lab = "λ_on_x_"
        end
    end

    if states == 0
        state_lab = ""
    elseif states == 1
        state_lab = "_two_state"
    elseif states == 2
        state_lab = "_two_and_three_state"
    end

    title = stat_type_lab * plot_lab * fixed_var * state_lab * ".png"
    return title
end

function load_data(data_folder, state = 0)
    if multiple_data == false
        matrix = zeros(length(lambdas), length(magnitudes), length(velocities));
    else
        matrix = zeros(length(lambdas), length(magnitudes), length(velocities), multiple_data_number);
    end
    for l_indx in 1:length(lambdas)
        for m_indx = 1:length(magnitudes)
            mag = magnitudes[m_indx]
            for n_indx = 1:length(velocities)              
                if data_folder == "data/u_0"
                    data_name = "mag_" * string(mag) * "_U_" * string(velocities[n_indx]) * "_lambda_" * string(lambdas[l_indx]) * ".jld2"
                    load_name = joinpath(data_folder, data_name)
                    try
                        @load load_name cs cf
                        var = obtain_stat(stat_type, cs, cf)
                    catch systemerror
                        print("no file named" * data_name)
                        var = 0
                    end
                elseif data_folder == "data/gpu/kappa_0.001/code_fixes/"
                    data_folder = "data/gpu/kappa_0.001/code_fixes/"
                    if κ == 0.0001
                        data_name = "mag_" * string(mag) * "_U_" * string(velocities[n_indx]) * "_lambda_" * string(lambdas[l_indx]) * "_k_0.001.jld2"
                    else
                        data_name = "mag_" * string(mag) * "_U_" * string(velocities[n_indx]) * "_lambda_" * string(lambdas[l_indx]) * ".jld2"
                    end
                    # Concatenate the folder and file name to get the full path
                    load_name = joinpath(data_folder, data_name)
                    try
                        @load load_name c_mean flux_mean c_squared_mean gc
                        var = obtain_stat_means(stat_type, c_mean, flux_mean, c_squared_mean, gc, mag, lambdas[l_indx], k)
                        if multiple_data == true
                            matrix[l_indx, m_indx, n_indx, :] = var
                        else
                            matrix[l_indx, m_indx, n_indx] = var
                        end
    
                    catch systemerror
                    var = 0
                        print("no file named " * data_name)
                    end              
                else
                    data_name = "mag_" * string(mag) * "_U_" * string(velocities[n_indx]) * "_lambda_" * string(lambdas[l_indx]) * "_k_" * string(κ) * "_N_" * string(state) * ".jld2"
                    load_name = joinpath(data_folder, data_name)
                    try
                        @load load_name cs
                        c_mean = sum(cs[:, :, end], dims = 2)
                        flux_mean = []
                        c_squared_mean = []
                        gc=real(ifft(im*k[:,1].*fft(c_mean)));
                        var = obtain_stat_means(stat_type, c_mean, flux_mean, c_squared_mean, gc, mag, lambdas[l_indx], k)
                        if multiple_data == true
                            matrix[l_indx, m_indx, n_indx, :] = var
                        else
                            matrix[l_indx, m_indx, n_indx] = var
                        end                        
                    catch systemerror
                        var = 0
                            print("no file named " * data_name)
                    end
                end
                #print("n_indx=" * string(n_indx))
                
            end
        end
    end
    if stat_type == 17
        for m_indx = 1:length(magnitudes)
            for n_indx = 1:length(velocities) 
                matrix[:, m_indx, n_indx] = matrix[:, m_indx, n_indx]/maximum(matrix[:, m_indx, n_indx])
            end
        end
    end
    
    matrix = replace(matrix) do x
        x == 0 ? NaN : x
    end
    return matrix
end

function plot_data(leg_type)
    # plot the data
    if plot_type == 1 # Δ on x
        if line_variable == 2
            if !multiple_data
                for indx = 1:length(velocities)
                    colours = specify_colours(length(velocities))
                    scatterlines!(magnitudes, matrix[lambda_or_kappa_choice_indx, :, indx], label = leg_type * "U = " * string(velocities[indx]), color = colours[indx]; line_options...)
                end
            else
                if data_line_colours == 2
                    if length(velocities) == 1
                        colours = specify_colours(length(velocities)+1)
                    else                    
                        colours = specify_colours(length(velocities))
                    end
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(magnitudes, matrix[lambda_or_kappa_choice_indx, :, indx, mul_indx], label = leg_type * "U = " * string(velocities[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(magnitudes, matrix[lambda_or_kappa_choice_indx, :, indx, mul_indx], label = leg_type * "U = " * string(velocities[indx]) * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end
            end
        elseif line_variable == 4
            if varu == 0
                lab_start = "κ = "
            else
                lab_start = "λ = "
            end
            if !multiple_data
                for indx = 1:length(lambdas)
                    colours = specify_colours(length(lambdas))
                    scatterlines!(magnitudes, matrix[indx, :, vel_choice_indx], label = leg_type * lab_start * string(lambdas[indx]), color = colours[indx]; line_options...)
                end
            else            
                if data_line_colours == 2
                    if length(lambdas) == 1
                        colours = specify_colours(length(lambdas)+1)
                    else                    
                        colours = specify_colours(length(lambdas))
                    end
                    for indx = 1:length(lambdas)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(magnitudes, matrix[indx, :, vel_choice_indx, mul_indx], label = leg_type * lab_start * string(lambdas[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(lambdas)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(magnitudes, matrix[indx, :, vel_choice_indx, mul_indx], label = leg_type * lab_start * string(lambdas[indx])  * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end


            end    
        ax.xlabel = "|Δ|"
        end
    elseif plot_type == 2 # U on x
        if line_variable == 1
            if !multiple_data
                for indx = 1:length(magnitudes)
                    colours = specify_colours(length(magnitudes))
                    scatterlines!(velocities, matrix[lambda_or_kappa_choice_indx, indx, :], label = leg_type * "|Δ| = " * string(magnitudes[indx]), color = colours[indx]; line_options...)
                end
            else
                if data_line_colours == 2
                    if length(magnitudes) == 1
                        colours = specify_colours(length(magnitudes)+1)
                    else                    
                        colours = specify_colours(length(magnitudes))
                    end
                    for indx = 1:length(magnitudes)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(velocities, matrix[lambda_or_kappa_choice_indx, indx, :, mul_indx], label = leg_type *  "|Δ| = " * string(magnitudes[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(velocities, matrix[lambda_or_kappa_choice_indx, indx, :, mul_indx], label = leg_type * "|Δ| = " * string(magnitudes[indx]) * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end 
            end

        elseif line_variable == 4
            if varu == 0
                lab_start = "κ = "
            else
                lab_start = "λ = "
            end
            if !multiple_data
                for indx = 1:length(lambdas)
                    colours = specify_colours(length(lambdas))
                    scatterlines!(velocities, matrix[indx, mag_choice_indx, :], label = leg_type * lab_start * string(lambdas[indx]), color = colours[indx]; line_options...)
                end
            else
                if data_line_colours == 2
                    if length(magnitudes) == 1
                        colours = specify_colours(length(magnitudes)+1)
                    else                    
                        colours = specify_colours(length(magnitudes))
                    end
                    for indx = 1:length(magnitudes)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(velocities, matrix[indx, mag_choice_indx, :, mul_indx], label = leg_type * lab_start * string(lambdas[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(velocities, matrix[indx, mag_choice_indx, :, mul_indx], label = leg_type * lab_start * string(lambdas[indx]) * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end             
            end    
        end
        ax.xlabel = "U"
    else # λ or κ on x
        if line_variable == 1 # Δ for each line
            if !multiple_data        
                for indx = 1:length(magnitudes)
                    if length(magnitudes) == 1
                        colours = specify_colours(2)
                    else
                        colours = specify_colours(length(magnitudes))
                    end
                    scatterlines!(lambdas, matrix[:, indx, vel_choice_indx], label = leg_type * "|Δ| = " * string(magnitudes[indx]), color = colours[indx]; line_options...) 
                end   
            else
                if data_line_colours == 2
                    if length(magnitudes) == 1
                        colours = specify_colours(length(magnitudes)+1)
                    else                    
                        colours = specify_colours(length(magnitudes))
                    end
                    for indx = 1:length(magnitudes)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(lambdas, matrix[:, indx, vel_choice_indx, mul_indx], label = leg_type * "|Δ| = " * string(magnitudes[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(magnitudes)
                        for mul_indx = 1:multiple_data_number
                            print("indx", indx, "mul_indx", mul_indx)
                        scatterlines!(lambdas, matrix[:, indx, vel_choice_indx, mul_indx], label = leg_type * "|Δ| = " * string(magnitudes[indx]) * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end             
            end

        elseif line_variable ==2 # U for each line
            if !multiple_data
                for indx = 1:length(velocities)
                    if length(velocities) == 1
                        colours = specify_colours(2)
                    else
                        colours = specify_colours(length(velocities))
                    end
                    scatterlines!(lambdas, matrix[:, mag_choice_indx, indx], label = leg_type * "U = " * string(velocities[indx]), color = colours[indx]; line_options...)
                end
            else
                if data_line_colours == 2
                    if length(velocities) == 1
                        colours = specify_colours(length(velocities)+1)
                    else                    
                        colours = specify_colours(length(velocities))
                    end
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(lambdas, matrix[:, mag_choice_indx, indx, mul_indx], label = leg_type * "U = " * string(velocities[indx]) * data_labels[mul_indx], color = colours[indx], marker = markers[mul_indx]; line_options...)
                        end
                    end   
                else
                    colours = specify_colours(multiple_data_number)
                    for indx = 1:length(velocities)
                        for mul_indx = 1:multiple_data_number
                        scatterlines!(lambdas, matrix[:, mag_choice_indx, indx, mul_indx], label = leg_type *  "U = " * string(velocities[indx]) * data_labels[mul_indx], color = colours[mul_indx], marker = markers[indx]; line_options...)
                        end
                    end 
                end            
            end
        else
            for mag_indx = 1:length(magnitudes)
                colours = specify_colours(length(velocities))
                for k_indx = 1:length(velocities)
                    scatterlines!(lambdas, matrix[:, mag_choice_indx, k_indx], label = leg_type *  "U = " * string(velocities[k_indx]) *  ", |Δ| = " * string(magnitudes[mag_indx]), color = colours[k_indx], marker = markers[mag_indx]; line_options...)
                end
            end
        end
        if varu == 0
            ax.xlabel = "κ"
        else
            ax.xlabel = "λ"
        end
    end
    return nothing
end


fig = Figure(resolution = (3024, 1964),
xlabelsize = 22, ylabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = -10)
four_panels = [1, 1, 2, 2, 1, 2, 1, 2]
if plot_type == 3 #&& varu == 0
    if stat_type == 1 || stat_type == 3
        ax = Axis(fig[1, 1], xscale = log10) #, yscale = log10)
    else
        ax = Axis(fig[1, 1], xscale = log10)
    end
elseif stat_type ==3
    ax = Axis(fig[1, 1], yscale = log10)
else
    ax = Axis(fig[1, 1]) #, yscale = log10)
    #ax = Axis(fig[1, 1], yscale = log10)
end

data_folder = "data/gpu/kappa_0.001/code_fixes/"
matrix = load_data(data_folder)
leg_type = ""
line_options = (; linewidth = 4, markersize = 50)
plot_data(leg_type)
indx = 1
for state in states_to_add
    indx = indx+1
    data_folder = "data/gpu/kappa_0.001/" * string(state) * "_state/"
    matrix = load_data(data_folder, state)
    leg_type = string(state) * " state, "
    line_options = (; linewidth = 4, markersize = 50, marker = markers[indx])
    plot_data(leg_type)
end

#matrix = load_data(data_folder, 2)
##data_folder = "data/gpu/kappa_0.001/test_2_state/"
#leg_type = "three state, "
#leg_type = "test_two_state, "
#line_options = (; linewidth = 4, markersize = 50, marker = markers[4])
#plot_data(leg_type)

# load in the data
#data_folder = "data/gpu/kappa_0.001/code_fixes/"
#matrix = load_data(data_folder)
###if num_states_to_add == 0
#    leg_type = ""
#    line_options = (; linewidth = 4, markersize = 50)
#    plot_data(leg_type)
#elseif num_states_to_add == 1
#    leg_type = ""
#    data_folder = "data/gpu/kappa_0.001/code_fixes/"
#    matrix = load_data(data_folder)
#    line_options = (; linewidth = 4, markersize = 50)
#    plot_data(leg_type)
#    data_folder = "data/gpu/kappa_0.001/two_state/"
#    matrix = load_data(data_folder)
#    leg_type = "two state, "
#    line_options = (; linewidth = 4, markersize = 50, marker = markers[2])
#    plot_data(leg_type)
#elseif num_states_to_add == 2
#    leg_type = ""
#    data_folder = "data/gpu/kappa_0.001/code_fixes/"
#    matrix = load_data(data_folder)
#    line_options = (; linewidth = 4, markersize = 50)
#    plot_data(leg_type)
#    data_folder = "data/gpu/kappa_0.001/two_state/"
#    matrix = load_data(data_folder)
#    leg_type = "two state, "
#    line_options = (; linewidth = 4, markersize = 50, marker = markers[2])
#    plot_data(leg_type)
#    data_folder = "data/gpu/kappa_0.001/three_state/"
#    data_folder = "data/gpu/kappa_0.001/test_two_state/"
#    matrix = load_data(data_folder)
#    leg_type = "three state, "
#    leg_type = "test_two_state, "
#    line_options = (; linewidth = 4, markersize = 50, marker = markers[3])
#    plot_data(leg_type)
#end


if line_variable !=3
    if plot_type !=1 && line_variable !=1
        ax.title = "|Δ| = " * string(magnitudes[mag_choice_indx])
    elseif plot_type !=2 && line_variable !=2
        ax.title = "U = " * string(velocities[vel_choice_indx])
    else
        if varu == 0
            ax.title = "κ = " * string(lambdas[lambda_or_kappa_choice_indx])
        else
            ax.title = "λ = " * string(lambdas[lambda_or_kappa_choice_indx])
        end
    end
end

line_options = (; linewidth = 4, markersize = 50)
if stat_type == 4 && plot_type != 1
    mag = magnitudes[mag_choice_indx]
    hlines!(ax, 1, color = :blue, label = L"c_0 = 1"; line_options...)
    hlines!(ax, sqrt(1-mag^2), color = :red, label = L"c_0 = \sqrt{1-|Δ|^2}" ; line_options...)
    hlines!(ax, 1/2 + sqrt(1-mag^2)/2, color = :green, label = L"c_0 = \frac{1+\sqrt{1-|Δ|^2}}{2}" ; line_options...)
elseif stat_type == 1 && plot_type != 1
    mag = magnitudes[mag_choice_indx]
    hlines!(ax, 0, color = :blue, label = L"range = 0"; line_options...)
    hlines!(ax, 2*mag, color = :red, label = L"range = 2|Δ|" ; line_options...)
    hlines!(ax, mag, color = :green, label = L"range = |Δ|" ; line_options...)
end

ax.titlesize = axtitlesize
ax.ylabel = axis_labels(stat_type)
ax.xlabelsize = axlabelsize
ax.ylabelsize = axlabelsize
ax.xticklabelsize = axticksize
ax.yticklabelsize = axticksize

legend = axislegend(labelsize = legendsize, nbanks = 1)

save(axis_title(stat_type, plot_type, num_states_to_add), fig)