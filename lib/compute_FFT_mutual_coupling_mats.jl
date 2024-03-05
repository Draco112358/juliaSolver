include("compute_Circulant_P_sup.jl")
include("compute_Circulant_Lp.jl")

function compute_FFT_mutual_coupling_mats(circulant_centers, escalings, Nx, Ny, Nz, QS_Rcc_FW)
    FFTCP, FFTCLp = nothing, nothing
    if QS_Rcc_FW == 1
        FFTCP = compute_Circulant_P_sup(circulant_centers, escalings, Nx, Ny, Nz)
        FFTCLp = compute_Circulant_Lp(circulant_centers, escalings, Nx, Ny, Nz)
        
    elseif QS_Rcc_FW == 2
        distance_method = "RCC" # RCC AVG MIN
        FFTCP = compute_rows_Rcc_P(circulant_centers, distance_method)
        FFTCLp = compute_rows_Rcc_Lp(circulant_centers, distance_method)
    else
        FFTCP = compute_rows_Taylor_P(circulant_centers)
        FFTCLp = compute_rows_Taylor_Lp(circulant_centers)
    end
    return FFTCP, FFTCLp
end
