include("compute_Lp_Voxels.jl")
include("From_1D_to_3D.jl")

using FFTW

function compute_Circulant_Lp(circulant_centers,escalings,Nx,Ny,Nz)
    
    FFTW.set_num_threads(12)
    
    enable_accuracy_Lp=0
    
    #println("Lp computation started")
    escaling=escalings["Lp"]
    sx=circulant_centers["sx"]
    sy=circulant_centers["sy"]
    sz=circulant_centers["sz"]
    FFTCLp=Array{Any}(undef, 3, 2)
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
        FFTCLp[1,1]=zeros(0,2*Ny,2*Nz)
        FFTCLp[1,2]=zeros(0,2*Ny,2*Nz)
    end
    dc=2
    if Nly>0
        Lpy_row=escaling*compute_Lp_Voxels(circulant_centers["Lpy"][1,:],circulant_centers["Lpy"],sx,sy,sz,sx,sy,sz,dc,is_sym)
        Lpy_row[1]=0
        FFTCLp[2,1]=fft(store_circulant(Lpy_row,Nx,Ny-1,Nz))
    else
        FFTCLp[2,1]=zeros(2*Nx,0,2*Nz)
        FFTCLp[2,2]=zeros(2*Nx,0,2*Nz)
    end
    dc=3
    if Nlz>0
        Lpz_row=escaling*compute_Lp_Voxels(circulant_centers["Lpz"][1,:],circulant_centers["Lpz"],sx,sy,sz,sx,sy,sz,dc,is_sym)
        Lpz_row[1]=0
        FFTCLp[3,1]=fft(store_circulant(Lpz_row,Nx,Ny,Nz-1))
    else
        FFTCLp[3,1]=zeros(2*Nx,2*Ny,0)
        FFTCLp[3,2]=zeros(2*Nx,2*Ny,0)
    end
    # println("Lp computation ended. Elapsed time = ")
    return FFTCLp
end

function store_circulant(row_Lp,Nx,Ny,Nz)
    i1x=1:Nx
    i2x=Nx+2:2*Nx
    i3x=Nx:-1:2
    i1y=1:Ny
    i2y=Ny+2:2*Ny
    i3y=Ny:-1:2
    i1z=1:Nz
    i2z=Nz+2:2*Nz
    i3z=Nz:-1:2
    Circ=zeros(Nx,Ny,Nz)
    for cont in range(1,length(row_Lp))
        m,n,k=From_1D_to_3D(Nx, Ny, cont)
        Circ[m,n,k]=row_Lp[cont]
    end
    Cout=zeros(2*Nx,2*Ny,2*Nz)
    Cout[i1x,i1y,i1z]=Circ[i1x,i1y,i1z]
    Cout[i2x,i1y,i1z]=Circ[i3x,i1y,i1z]
    Cout[i1x,i2y,i1z]=Circ[i1x,i3y,i1z]
    Cout[i1x,i1y,i2z]=Circ[i1x,i1y,i3z]
    Cout[i2x,i2y,i1z]=Circ[i3x,i3y,i1z]
    Cout[i2x,i1y,i2z]=Circ[i3x,i1y,i3z]
    Cout[i1x,i2y,i2z]=Circ[i1x,i3y,i3z]
    Cout[i2x,i2y,i2z]=Circ[i3x,i3y,i3z]

    return Cout
end
