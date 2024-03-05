include("compute_row_P_sup.jl")
include("From_1D_to_3D.jl")

using FFTW

function compute_Circulant_P_sup(circulant_centers,escalings,Nx,Ny,Nz)
    #println("P computation started")
    FFTW.set_num_threads(12)
    
    sx = circulant_centers["sx"]
    sy = circulant_centers["sy"]
    sz = circulant_centers["sz"]
    FFTCP = Array{Any}(undef, 3, 3)
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p12_se"][1,:], circulant_centers["p12_se"], sx, sy, sz, 1, 1)
    FFTCP[1,1] = fft(store_circulant(row_P, Nx, Ny+1, Nz))
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p34_se"][1,:], circulant_centers["p34_se"], sx, sy, sz, 3, 3)
    FFTCP[2,2] = fft(store_circulant(row_P, Nx+1, Ny, Nz))
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p56_se"][1,:], circulant_centers["p56_se"], sx, sy, sz, 5, 5)
    FFTCP[3,3] = fft(store_circulant(row_P, Nx, Ny, Nz+1))
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p1234"][1,:], circulant_centers["p1234"], sx, sy, sz, 3, 2)
    FFTCP[1,2] = fft(store_circulant(row_P, 2*(Nx+1)-1, 2*(Ny+1)-1, Nz))
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p1256"][1,:], circulant_centers["p1256"], sx, sy, sz, 5, 2)
    FFTCP[1,3] = fft(store_circulant(row_P, Nx, 2*(Ny+1)-1, 2*(Nz+1)-1))
    row_P = escalings["P"] * compute_row_P_sup(circulant_centers["p3456"][1,:], circulant_centers["p3456"], sx, sy, sz, 5, 4)
    FFTCP[2,3] = fft(store_circulant(row_P, 2*(Nx+1)-1, Ny, 2*(Nz+1)-1))
    # println("P computation ended. Elapsed time = ")
    return FFTCP
end

function store_circulant(row_P, Nx, Ny, Nz)
    i1x = 1:Nx
    i2x = Nx+2:2*Nx
    i3x = Nx:-1:2
    i1y = 1:Ny
    i2y = Ny+2:2*Ny
    i3y = Ny:-1:2
    i1z = 1:Nz
    i2z = Nz+2:2*Nz
    i3z = Nz:-1:2
    Circ = zeros(Nx, Ny, Nz)
    for cont in range(1, length(row_P))
        m, n, k = From_1D_to_3D(Nx, Ny, cont)
        Circ[m, n, k] = row_P[cont]
    end
    Cout = zeros(2*Nx, 2*Ny, 2*Nz)
    Cout[i1x, i1y, i1z] = Circ[i1x, i1y, i1z]
    Cout[i2x, i1y, i1z] = Circ[i3x, i1y, i1z]
    Cout[i1x, i2y, i1z] = Circ[i1x, i3y, i1z]
    Cout[i1x, i1y, i2z] = Circ[i1x, i1y, i3z]
    Cout[i2x, i2y, i1z] = Circ[i3x, i3y, i1z]
    Cout[i2x, i1y, i2z] = Circ[i3x, i1y, i3z]
    Cout[i1x, i2y, i2z] = Circ[i1x, i3y, i3z]
    Cout[i2x, i2y, i2z] = Circ[i3x, i3y, i3z]
    return Cout
end
