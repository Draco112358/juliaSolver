include("prod_real_complex.jl")
using MKL

function ComputeMatrixVector(x::Array{ComplexF64}, w::Float64, incidence_selection::Dict, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chiVector, chi2Vector)
    m = size(incidence_selection["A"], 1)
    ns = size(incidence_selection["Gamma"], 2)
    I = @view x[1:m]
    Q = @view x[m+1:m+ns]
    Phi = @view x[m+ns+1:end]
    resProd = zeros(ComplexF64, size(expansions["exp_P"][3, 2], 1), 1)
    # Lp * I ---------------------------------------------------------------
    mx = incidence_selection["mx"]
    my = incidence_selection["my"]
    mz = incidence_selection["mz"]
    Y1 = zeros(ComplexF64, m)
    ind_aux_Lp = Dict(1 => 1:mx, 2 => mx+1:mx+my, 3 => mx+my+1:mx+my+mz)
    for cont = 1:3
        Nx = size(FFTCLp[cont, 1], 1) ÷ 2
        Ny = size(FFTCLp[cont, 1], 2) ÷ 2
        Nz = size(FFTCLp[cont, 1], 3) ÷ 2
        Ired = I[ind_aux_Lp[cont]]
        I_exp = @view resProd[1:size(expansions["mat_map_Lp"][cont, 1],1),1]
        mul!(I_exp, expansions["mat_map_Lp"][cont, 1], Ired)
        
        #I_exp = prod_real_complex(expansions["mat_map_Lp"][cont, 1], Ired)
        # I_exp = similar(Ired)
        # mul!(I_exp, expansions["mat_map_Lp"][cont, 1], Ired)
        #I_exp = bmul(expansions["mat_map_Lp"][cont, 1] , Ired)
        CircKT = reshape(I_exp, Nx, Ny, Nz)
        padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
        padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
        Chi = customIfftOptimized(PLIVector[cont], PVector[cont], padded_CircKt, FFTCLp[cont,1], chiVector[cont])
        Y1[ind_aux_Lp[cont]] = Y1[ind_aux_Lp[cont]] + prod_real_transposed_complex(expansions["mat_map_Lp"][cont, 1], reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1))
    end
    Y1 =   lmul!(1im * w,Y1) + DZ .* I + prod_real_complex(incidence_selection["A"] , Phi)
    
    # ---------------- P * Q ---------------------------------------------
    Y2 = zeros(ComplexF64,ns)
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz = size(FFTCP[cont1, cont2], 3) ÷ 2
            Q_exp = prod_real_complex(expansions["exp_P"][cont1, cont2], Q)
            CircKT = reshape(Q_exp, Nx, Ny, Nz)
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
            Chi = customIfftOptimized(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
            Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont2, cont1], (reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            if cont1 != cont2
                Q_exp = prod_real_complex(expansions["exp_P"][cont2, cont1], Q)
                CircKT = reshape(Q_exp, Nx, Ny, Nz)
                padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
                Chi = customIfftOptimized(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
                Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont1, cont2], (reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            end
        end
    end
    Y2 = Y2 - prod_real_transposed_complex(incidence_selection["Gamma"] , Phi)
    Y3 = lmul!(-1.0, prod_real_transposed_complex(incidence_selection["A"] , I)) + prod_real_complex(Yle , Phi) +  lmul!(1im * w, prod_real_complex(incidence_selection["Gamma"], Q))
    MatrixVector = precond_3_3_vector(lu, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], w, Y1, Y2, Y3)
    
    return MatrixVector    
end


function precond_3_3_vector(F,invZ,invP,A,Gamma,w,X1,X2,X3)
    n1=length(X1)
    n2=length(X2)
    n3=length(X3)

    i1=range(1, stop=n1)
    i2=range(n1+1,stop=n1+n2)
    i3=range(n1+n2+1,stop=n1+n2+n3)

    Y=zeros(ComplexF64 , n1+n2+n3)
    
    
    M1 = prod_real_complex(invZ, X1)
   
    M2 = F\(prod_real_transposed_complex(A, M1))  
    
    M3 = prod_real_complex(invP, X2)  
    
    M4 = F\(prod_real_complex(Gamma, M3)) 
    
    M5 = F\X3
    

    Yi1 = @view Y[i1]   
    Y[i1] .= Yi1 .+ M1-1.0*(prod_real_complex(invZ,prod_real_complex(A, M2))) .+ lmul!(1im*w, prod_real_complex(invZ,prod_real_complex(A, M4))) .- lmul!(1.0,prod_real_complex(invZ,prod_real_complex(A, M5)))

    Yi2 = @view Y[i2]
    Y[i2] .= Yi2 .+ (prod_real_complex(invP,prod_real_transposed_complex(Gamma, M2))) .+ M3 - lmul!(1im*w,prod_real_complex(invP,prod_real_transposed_complex(Gamma, M4))) .+ (prod_real_complex(invP,prod_real_transposed_complex(Gamma, M5)))
    
    Yi3 = @view Y[i3]
    Y[i3] .= Yi3 .+ M2 .- lmul!(1im*w,M4) .+ M5

    # return  convert(Array{ComplexF64}, Y)
    return Y
end


function customIfftOptimized(PLIVector, PVector, padded_CircKt, FFTCLp, chiVector)
    mul!(chiVector, PVector, padded_CircKt)
    b = FFTCLp .* chiVector
    mul!(chiVector, PLIVector, b)
end