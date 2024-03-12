using MKL

function ComputeMatrixVector(x::Array{ComplexF64}, w::Float64, incidence_selection::Dict, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector)
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
        CircKT = reshape(I_exp, Nx, Ny, Nz)
        padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
        padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
        fft_and_ifft_both_in_place!(PLIVector[cont], PVector[cont], padded_CircKt, FFTCLp[cont,1])
        mat_map_lp_view = @view resProd[1:size(expansions["mat_map_Lp"][cont, 1], 2), 1]
        mul!(mat_map_lp_view, transpose(expansions["mat_map_Lp"][cont, 1]), reshape(padded_CircKt[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz))
        @views Y1[ind_aux_Lp[cont]] = Y1[ind_aux_Lp[cont]] + mat_map_lp_view
    end
    A_view = @view resProd[1:size(incidence_selection["A"],1),1]
    mul!(A_view, incidence_selection["A"], Phi)
    Y1 =   lmul!(1im * w,Y1) + DZ .* I + A_view
    
    # ---------------- P * Q ---------------------------------------------
    Y2 = zeros(ComplexF64,ns)
    Y3 = zeros(ComplexF64,size(Yle,1))
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz = size(FFTCP[cont1, cont2], 3) ÷ 2
            Q_exp = @view resProd[1:size(expansions["exp_P"][cont1, cont2], 1), 1]
            mul!(Q_exp, expansions["exp_P"][cont1, cont2], Q)
            CircKT = reshape(Q_exp, Nx, Ny, Nz)
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
            Q_exp = @view resProd[1:size(expansions["exp_P"][cont2, cont1], 2), 1]
            if cont1 == cont2
                fft_and_ifft_both_in_place!(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2])
                mul!(Q_exp, transpose(expansions["exp_P"][cont2, cont1]), reshape(padded_CircKt[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz))
            else
                fft_and_in_place_ifft!(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
                mul!(Q_exp, transpose(expansions["exp_P"][cont2, cont1]), reshape(chi2Vector[cont1, cont2][1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz))
            end
            Y2 .= Y2 + Q_exp
            if cont1 != cont2
                Q_exp = @view resProd[1:size(expansions["exp_P"][cont2, cont1], 1), 1]
                mul!(Q_exp, expansions["exp_P"][cont2, cont1], Q)
                CircKT = reshape(Q_exp, Nx, Ny, Nz)
                padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
                fft_and_in_place_ifft!(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
                Q_exp = @view resProd[1:size(expansions["exp_P"][cont1, cont2], 2), 1]
                mul!(Q_exp, transpose(expansions["exp_P"][cont1, cont2]), reshape(chi2Vector[cont1, cont2][1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz))
                Y2 .= Y2 + Q_exp
            end
        end
    end
    Gamma_view = @view resProd[1:size(incidence_selection["Gamma"], 2), 1]
    mul!(Gamma_view, transpose(incidence_selection["Gamma"]), Phi)
    Y2 .= Y2 + lmul!(-1.0, Gamma_view)
    A_v2 = @view resProd[1:size(incidence_selection["A"], 2), 1]
    mul!(A_v2, transpose(incidence_selection["A"]), I)
    Y3 .= Y3 + lmul!(-1.0, A_v2) 
    Yle_view = @view resProd[1:size(Yle,1), 1]
    mul!(Yle_view, Yle, Phi)
    Y3 .= Y3 + Yle_view 
    Gamma_v2 = @view resProd[1:size(incidence_selection["Gamma"],1), 1]
    mul!(Gamma_v2, incidence_selection["Gamma"], Q)
    Y3 .= Y3 +  lmul!(1im * w, Gamma_v2)
    MatrixVector = precond_3_3_vector(lu, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], w, vec(Y1), vec(Y2), vec(Y3), resProd)
    
    return MatrixVector    
end


function precond_3_3_vector(F,invZ,invP,A,Gamma,w,X1,X2,X3, resProd)

    n1=length(X1)
    n2=length(X2)
    n3=length(X3)

    i1=range(1, stop=n1)
    i2=range(n1+1,stop=n1+n2)
    i3=range(n1+n2+1,stop=n1+n2+n3)

    Y=zeros(ComplexF64 , n1+n2+n3)
    
    invZ_view = @view resProd[1:size(invZ,1),1]
    mul!(invZ_view, invZ, X1)
    Yi1 = @view Y[i1]
    Y[i1] .= Yi1 .+ invZ_view
   
    A_view = @view resProd[size(resProd,1)-size(A,2)+1:end,1]
    mul!(A_view, transpose(A), invZ_view)
    M2 = F\A_view
    
    invP_view = @view resProd[1:size(invP,1),1]
    mul!(invP_view, invP, X2)
    Yi2 = @view Y[i2]
    Y[i2] .= Yi2 .+ invP_view
    
    Gamma_view = @view resProd[size(resProd,1)-size(Gamma,1)+1:end,1]
    mul!(Gamma_view, Gamma, invP_view)
    M4 = F\Gamma_view
    
    M5 = F\X3
    

    # Yi1 = @view Y[i1]  
    A_view = @view resProd[1:size(A,1),1]
    mul!(A_view, A, M2)
    invZ_view = @view resProd[size(resProd,1)-size(invZ,1)+1:end,1]
    mul!(invZ_view, invZ, A_view)
    Y[i1] .= Yi1 .-lmul!(1.0, invZ_view)
    mul!(A_view, A, M4)
    mul!(invZ_view, invZ, A_view)
    Y[i1] .= Yi1 .+ lmul!(1im*w, invZ_view) 
    mul!(A_view, A, M5)
    mul!(invZ_view, invZ, A_view)
    Y[i1] .= Yi1 .- lmul!(1.0,invZ_view) 

    # Yi2 = @view Y[i2]
    Gamma_view = @view resProd[size(resProd,1)-size(Gamma,2)+1:end,1]
    mul!(Gamma_view, transpose(Gamma), M2)
    mul!(invP_view, invP, Gamma_view)
    Y[i2] .= Yi2 .+ invP_view 
    mul!(Gamma_view, transpose(Gamma), M4)
    mul!(invP_view, invP, Gamma_view)
    Y[i2] .= Yi2 .- lmul!(1im*w,invP_view)
    mul!(Gamma_view, transpose(Gamma), M5)
    mul!(invP_view, invP, Gamma_view) 
    Y[i2] .= Yi2 .+ invP_view
    
    Yi3 = @view Y[i3]
    Y[i3] = Yi3 .+ M2 .- lmul!(1im*w,M4) .+ M5

    return Y
end


function fft_and_in_place_ifft!(PLIVector, PVector, padded_CircKt, FFTCLp, chiVector)
    mul!(chiVector, PVector, padded_CircKt)
    chiVector .= FFTCLp .* chiVector
    PLIVector*chiVector
    return chiVector
end

function fft_and_ifft_both_in_place!(PLIVector, PVector, padded_CircKt, FFTCLp)
    PVector*padded_CircKt
    padded_CircKt .= FFTCLp .* padded_CircKt
    PLIVector*padded_CircKt
    return padded_CircKt
end