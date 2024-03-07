include("compute_Lp_Voxels.jl")
include("compute_row_P_sup.jl")
using FFTW

function compute_FFT_mutual_coupling_mats(circulant_centers, escalings, Nx, Ny, Nz, QS_Rcc_FW)
    FFTW.set_num_threads(12)
    FFTW.set_provider!("mkl")
    # FFTCP, FFTCLp = Array{Array{ComplexF64}}(undef, 3, 3), nothing
    # if QS_Rcc_FW == 1
    FFTCP = compute_Circulant_P_sup(circulant_centers, escalings, Nx, Ny, Nz)
    FFTCLp = compute_Circulant_Lp(circulant_centers, escalings, Nx, Ny, Nz)

    # elseif QS_Rcc_FW == 2
    #     distance_method = "RCC" # RCC AVG MIN
    #     FFTCP = compute_rows_Rcc_P(circulant_centers, distance_method)
    #     FFTCLp = compute_rows_Rcc_Lp(circulant_centers, distance_method)
    # else
    #     FFTCP = compute_rows_Taylor_P(circulant_centers)
    #     FFTCLp = compute_rows_Taylor_Lp(circulant_centers)
    # end
    return FFTCP, FFTCLp
end


function compute_Circulant_Lp(circulant_centers,escalings,Nx,Ny,Nz)
    
    enable_accuracy_Lp=0
    
    #println("Lp computation started")
    escaling=escalings["Lp"]
    sx=circulant_centers["sx"]
    sy=circulant_centers["sy"]
    sz=circulant_centers["sz"]
    FFTCLp=Array{Array{ComplexF64}}(undef, 3, 2)
    is_sym=false
    dc=1
    Nlx=size(circulant_centers["Lpx"],1)
    Nly=size(circulant_centers["Lpy"],1)
    Nlz=size(circulant_centers["Lpz"],1)
    if Nlx>0
        Lpx_row=escaling*compute_Lp_Voxels(circulant_centers["Lpx"][1,:],circulant_centers["Lpx"],sx,sy,sz,sx,sy,sz,dc,is_sym)
        Lpx_row[1]=0
        FFTCLp[1,1]=fft(store_circulant(Lpx_row,Nx-1,Ny,Nz))
    else
        FFTCLp[1,1]=zeros(ComplexF64, 0, 2*Ny, 2*Nz)
        FFTCLp[1,2]=zeros(ComplexF64, 0, 2*Ny, 2*Nz)
    end
    dc=2
    if Nly>0
        Lpy_row=escaling*compute_Lp_Voxels(circulant_centers["Lpy"][1,:],circulant_centers["Lpy"],sx,sy,sz,sx,sy,sz,dc,is_sym)
        Lpy_row[1]=0
        FFTCLp[2,1]=fft(store_circulant(Lpy_row,Nx,Ny-1,Nz))
    else
        FFTCLp[2,1]=zeros(ComplexF64, 2*Nx,0,2*Nz)
        FFTCLp[2,2]=zeros(ComplexF64, 2*Nx,0,2*Nz)
    end
    dc=3
    if Nlz>0
        Lpz_row=escaling*compute_Lp_Voxels(circulant_centers["Lpz"][1,:],circulant_centers["Lpz"],sx,sy,sz,sx,sy,sz,dc,is_sym)
        Lpz_row[1]=0
        FFTCLp[3,1]=fft(store_circulant(Lpz_row,Nx,Ny,Nz-1))
    else
        FFTCLp[3,1]=zeros(ComplexF64, 2*Nx,2*Ny,0)
        FFTCLp[3,2]=zeros(ComplexF64, 2*Nx,2*Ny,0)
    end
    # println("Lp computation ended. Elapsed time = ")
    return FFTCLp
end

function compute_Circulant_P_sup(circulant_centers,escalings,Nx,Ny,Nz)
    #println("P computation started")
    
    sx = circulant_centers["sx"]
    sy = circulant_centers["sy"]
    sz = circulant_centers["sz"]
    FFTCP = Array{Array{ComplexF64}}(undef, 3, 3)
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
        m, n, k = from_1D_to_3D(Nx, Ny, cont)
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

function from_1D_to_3D( M, N, pos)
    pos=pos-1;
	k = floor(pos / (M * N));
	j = floor((pos - k* M * N) / M);
	i = mod(pos - k * M * N, M);
    k=k+1;
    j=j+1;
    i=i+1;
    return convert(Int64,i),convert(Int64,j),convert(Int64,k)
end