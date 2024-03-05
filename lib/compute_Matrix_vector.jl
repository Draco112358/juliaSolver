function ComputeMatrixVector(x, w, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, time1Vector, time2Vector)
    m = size(incidence_selection["A"], 1)
    ns = size(incidence_selection["Gamma"], 2)
    I = x[1:m]
    Q = x[m+1:m+ns]
    Phi = x[m+ns+1:end]
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
        I_exp = prod_real_complex(expansions["mat_map_Lp"][cont, 1] , Ired)
        # I_exp = similar(Ired)
        # mul!(I_exp, expansions["mat_map_Lp"][cont, 1], Ired)
        #I_exp = bmul(expansions["mat_map_Lp"][cont, 1] , Ired)
        CircKT = reshape(I_exp, Nx, Ny, Nz)
        padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
        padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
        Chi = customIfftOptimized(PLIVector, PVector, padded_CircKt, FFTCLp, cont)
        Y1[ind_aux_Lp[cont]] = Y1[ind_aux_Lp[cont]] + prod_real_transposed_complex(expansions["mat_map_Lp"][cont, 1] , reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1))
    end
    Y1 = 1im * w * Y1 + DZ .* I + prod_real_complex(incidence_selection["A"] , Phi)
    
    # ---------------- P * Q ---------------------------------------------
    Y2 = zeros(ComplexF64,ns)
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz = size(FFTCP[cont1, cont2], 3) ÷ 2
            Q_exp = prod_real_complex(expansions["exp_P"][cont1, cont2] , Q)
            CircKT = reshape(Q_exp, Nx, Ny, Nz)
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
            time1 = @elapsed Chi = customIfftOptimized2(PLI2Vector, P2Vector, padded_CircKt, FFTCP, cont1, cont2)
            push!(time1Vector, time1)
            Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont2, cont1] , (reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            if cont1 != cont2
                Q_exp = prod_real_complex(expansions["exp_P"][cont2, cont1] , Q)
                CircKT = reshape(Q_exp, Nx, Ny, Nz)
                padded_CircKt[1:size(CircKT,1), 1:size(CircKT,2), 1:size(CircKT,3)] = CircKT
                time2 = @elapsed Chi = customIfftOptimized2(PLI2Vector, P2Vector, padded_CircKt, FFTCP, cont1, cont2)
                push!(time2Vector, time2)
                Y2 = Y2 + prod_real_transposed_complex(expansions["exp_P"][cont1, cont2] , (reshape(Chi[1:Nx, 1:Ny, 1:Nz], Nx * Ny * Nz, 1)))
            end
        end
    end
    Y2 = Y2 - prod_real_transposed_complex(incidence_selection["Gamma"] , Phi)
    Y3 = -1.0*(prod_real_transposed_complex(incidence_selection["A"] , I)) + prod_real_complex(Yle , Phi) + 1im * w * (prod_real_complex(incidence_selection["Gamma"] , Q))
    MatrixVector = precond_3_3_vector(lu, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], w, Y1, Y2, Y3)
    
    return MatrixVector    
end