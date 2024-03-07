include("prod_real_complex.jl")
using MKL
using FFTW

function ComputeMatrixVector(x::Array{ComplexF64}, w::Float64, incidence_selection::Dict, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu)
    m = size(incidence_selection["A"], 1)
    ns = size(incidence_selection["Gamma"], 2)
    I = @view x[1:m]
    Q = @view x[m+1:m+ns]
    Phi = @view x[m+ns+1:end]
    resProd = zeros(ComplexF64, size(expansions["exp_P"][3, 2], 1),1)
    # Lp * I ---------------------------------------------------------------
    mx = incidence_selection["mx"]
    my = incidence_selection["my"]
    mz = incidence_selection["mz"]
    Y1 = zeros(ComplexF64, m)
    ind_aux_Lp = Dict(1 => 1:mx, 2 => mx+1:mx+my, 3 => mx+my+1:mx+my+mz)
    PlansFFT = Array{FFTW.cFFTWPlan{ComplexF64, -1, false, 3, Tuple{Int64, Int64, Int64}},2}(undef, 3, 3)
    PlansIFFT = Array{AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 3, UnitRange{Int64}}, Float64},2}(undef, 3, 3)
    res = Array{Any, 2}(undef, 3, 3)
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
        Chi = ifft(FFTCLp[cont,1] .* (fft(padded_CircKt)))
        # Chi = customIfft(padded_CircKt, FFTCLp[cont,1])
        # Chi = customIfftOptimized(PLIVector[cont], PVector[cont], padded_CircKt, FFTCLp[cont,1], chiVector[cont])
        Y1[ind_aux_Lp[cont]] = Y1[ind_aux_Lp[cont]] + prod_real_transposed_complex(expansions["mat_map_Lp"][cont, 1], reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1))
    end
    A_phi = @view resProd[1:size(incidence_selection["A"],1),1]
    mul!(A_phi, incidence_selection["A"] , Phi)
    Y1 =   lmul!(1im * w,Y1) + DZ .* I + A_phi
    
    # ---------------- P * Q ---------------------------------------------
    Y2 = zeros(ComplexF64,ns)
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz = size(FFTCP[cont1, cont2], 3) ÷ 2
            Q_exp = @view resProd[1:size(expansions["exp_P"][cont1, cont2],1),1]
            mul!(Q_exp, expansions["exp_P"][cont1, cont2], Q)
            # Q_exp = prod_real_complex(expansions["exp_P"][cont1, cont2], Q)
            CircKT = reshape(Q_exp, Nx, Ny, Nz)
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
            res[cont1, cont2] = similar(padded_CircKt)
            PlansFFT[cont1, cont2] = plan_fft(padded_CircKt)
            PlansIFFT[cont1, cont2] = plan_ifft!(FFTCP[cont1, cont2])
            mul!(res[cont1, cont2], PlansFFT[cont1, cont2], padded_CircKt)
            res[cont1, cont2] = FFTCP[cont1, cont2] .* res[cont1, cont2]
            mul!(res[cont1, cont2], PlansIFFT[cont1, cont2], res[cont1, cont2])
            # Chi = PlansIFFT[cont1, cont2]*(FFTCP[cont1, cont2].*(PlansFFT[cont1, cont2]*padded_CircKt))
            # Chi = customIfft(padded_CircKt, FFTCP[cont1, cont2], PlansFFT[cont1, cont2])
            # Chi = customIfft(padded_CircKt, FFTCP[cont1, cont2])
            # Chi = customIfftOptimized(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
            Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont2, cont1], (reshape(res[cont1, cont2][1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            if cont1 != cont2
                Q_exp = @view resProd[1:size(expansions["exp_P"][cont2, cont1],1),1]
                mul!(Q_exp, expansions["exp_P"][cont2, cont1], Q)
                # Q_exp = prod_real_complex(expansions["exp_P"][cont2, cont1], Q)
                CircKT = reshape(Q_exp, Nx, Ny, Nz)
                padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
                mul!(res[cont1, cont2], PlansFFT[cont1, cont2], padded_CircKt)
                res[cont1, cont2] = FFTCP[cont1, cont2] .* res[cont1, cont2]
                mul!(res[cont1, cont2], PlansIFFT[cont1, cont2], res[cont1, cont2])
                # Chi = PlansIFFT[cont1, cont2]*(FFTCP[cont1, cont2].*(PlansFFT[cont1, cont2]*padded_CircKt))
                # Chi = customIfft(padded_CircKt, FFTCP[cont1, cont2], PlansFFT[cont1, cont2])
                # Chi = customIfft(padded_CircKt, FFTCP[cont1, cont2])
                # Chi = customIfftOptimized(PLI2Vector[cont1,cont2], P2Vector[cont1,cont2], padded_CircKt, FFTCP[cont1,cont2], chi2Vector[cont1, cont2])
                Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont1, cont2], (reshape(res[cont1, cont2][1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            end
        end
    end
    Y2 = Y2 - prod_real_transposed_complex(incidence_selection["Gamma"] , Phi)
    Yle_phi = @view resProd[1:size(Yle,1),1]
    mul!(Yle_phi, Yle, Phi)
    Y3 = lmul!(-1.0, prod_real_transposed_complex(incidence_selection["A"] , I)) + Yle_phi
    Gamma_Q = @view resProd[1:size(incidence_selection["Gamma"],1),1]
    mul!(Gamma_Q, incidence_selection["Gamma"], Q)
    Y3 = Y3 +  lmul!(1im * w, Gamma_Q)
    # Y3 = lmul!(-1.0, prod_real_transposed_complex(incidence_selection["A"] , I)) + prod_real_complex(Yle , Phi) +  lmul!(1im * w, prod_real_complex(incidence_selection["Gamma"], Q))
    MatrixVector = precond_3_3_vector(lu, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], w, Y1, Y2, Y3, resProd)
    
    return MatrixVector    
end


function precond_3_3_vector(F,invZ,invP,A,Gamma,w,X1,X2,X3, resP)
    n1=length(X1)
    n2=length(X2)
    n3=length(X3)

    i1=range(1, stop=n1)
    i2=range(n1+1,stop=n1+n2)
    i3=range(n1+n2+1,stop=n1+n2+n3)

    Y=zeros(ComplexF64 , n1+n2+n3)

    
    

    # invZ_view = @view resP[1:size(invZ,1),1]
    # mul!(invZ_view, invZ, X1)
    # Yi1 = @view Y[i1]  
    # Y[i1] .= Yi1 .+ invZ_view # invZ_view ha i valori di M1
    M1 = prod_real_complex(invZ, X1)
   
    # M2 = F\(prod_real_transposed_complex(A, invZ_view)) 
    M2 = F\(prod_real_transposed_complex(A, M1))  
    
    # invP_view = @view resP[1:size(invP,1)]
    # mul!(invP_view, invP, X2)
    # Yi2 = @view Y[i2]
    # Y[i2] .= Yi2 .+ invP_view # invP_view ha i dati di M3
    M3 = prod_real_complex(invP, X2)  
    
    # Gamma_view = @view resP[1:size(Gamma,1)]
    # mul!(Gamma_view, Gamma, invP_view)
    # M4 = F\Gamma_view
    M4 = F\(prod_real_complex(Gamma, M3)) 
    
    M5 = F\X3


    # A_view = @view resP[1:size(A,1)]
    # mul!(A_view, A, M2)
    # mul!(invZ_view, invZ, A_view)
    # Yi1 = @view Y[i1]  
    # Y[i1] .= Yi1 -lmul!(1.0, invZ_view)
    # mul!(A_view, A, M4)
    # mul!(invZ_view, invZ, A_view)
    # Yi1 = @view Y[i1]  
    # Y[i1] .= Yi1 .+ lmul!(1im*w, invZ_view)
    # mul!(A_view, A, M5)
    # mul!(invZ_view, invZ, A_view)
    # Yi1 = @view Y[i1]  
    # Y[i1] .= Yi1 .- lmul!(1.0, invZ_view)
    Yi1 = @view Y[i1]   
    Y[i1] .= Yi1 .+ M1-1.0*(prod_real_complex(invZ,prod_real_complex(A, M2))) .+ lmul!(1im*w, prod_real_complex(invZ,prod_real_complex(A, M4))) .- lmul!(1.0,prod_real_complex(invZ,prod_real_complex(A, M5)))
     
    # mul!(invP_view, invP, prod_real_transposed_complex(Gamma, M2))
    # Yi2 = @view Y[i2]  
    # Y[i2] .= Yi2 .+ invP_view 
    # mul!(invP_view, invP, prod_real_transposed_complex(Gamma, M4))
    # Yi2 = @view Y[i2]  
    # Y[i2] .= Yi2 - lmul!(1im*w, invP_view)
    # mul!(invP_view, invP, prod_real_transposed_complex(Gamma, M5))
    # Yi2 = @view Y[i2]  
    # Y[i2] .= Yi2 .+ invP_view
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

function customIfft(padded_CircKt, FFTCLp, PlanFFT)
    Chi = ifft(FFTCLp .* (PlanFFT*padded_CircKt))
    # Chi = ifft(FFTCLp .* fft(padded_CircKt))
    return Chi
end