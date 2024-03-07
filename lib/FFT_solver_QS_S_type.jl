using MKL
using SparseArrays, IterativeSolvers, FFTW, LinearAlgebra, LinearMaps
include("prod_real_complex.jl")
include("build_Yle_S.jl")
include("compute_Z_self.jl")
include("gmres_custom.jl")

function FFT_solver_QS_S_type(freq, escalings, incidence_selection, FFTCP, FFTCLp, diagonals, ports, lumped_elements, expansions, GMRES_settings, Zs_info, QS_Rcc_FW)
    FFTW.set_num_threads(12)
    #BLAS.set_num_threads(convert(Int64,Base.Threads.nthreads()/2))
    LinearAlgebra.BLAS.set_num_threads(1)
    #@save "/Users/edgardovittoria/Downloads/fftw.jl" FFTCLp FFTCP
    

    # println(Base.Threads.nthreads())
    freq = freq .* escalings["freq"]
    # GMRES settings ----------------------------
    
    Inner_Iter = GMRES_settings["Inner_Iter"]
    Outer_Iter = GMRES_settings["Outer_Iter"]
    # -------------------------------------------
    m = size(incidence_selection["A"], 1)
    n = size(incidence_selection["A"], 2)
    ns = size(incidence_selection["Gamma"], 2)
    w = 2 .* pi .* freq
    nfreq = length(w)
    is = zeros(Float64, n, 1)
    S = zeros(ComplexF64, size(ports["port_nodes"], 1), size(ports["port_nodes"], 1), length(freq))
    Vrest = zeros(ComplexF64, m + n + ns, size(ports["port_nodes"], 1))
    invP = sparse(1:ns, 1:ns, 1 ./ diagonals["P"],ns,ns)
    R_chiusura = 50.0
    PVector = Array{FFTW.cFFTWPlan{ComplexF64, -1, false, 3, Tuple{Int64, Int64, Int64}}}(undef, 3)
    PLIVector = Array{AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, Float64}}(undef,3)
    time1Vector::Array{Float64} = []
    time2Vector::Array{Float64} = []
    for cont = 1:3
            Nx = size(FFTCLp[cont, 1], 1) ÷ 2
            Ny = size(FFTCLp[cont, 1], 2) ÷ 2
            Nz = size(FFTCLp[cont, 1], 3) ÷ 2
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            P = plan_fft(padded_CircKt, flags=FFTW.MEASURE)
            PVector[cont] = P
            PLIVector[cont] = plan_ifft(FFTCLp[cont, 1] .* (P*padded_CircKt), flags=FFTW.MEASURE)
    end
    P2Vector = Matrix{FFTW.cFFTWPlan{ComplexF64, -1, false, 3, Tuple{Int64, Int64, Int64}}}(undef,3,3)
    PLI2Vector = Matrix{AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, Float64}}(undef,3, 3)
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz = size(FFTCP[cont1, cont2], 3) ÷ 2
            padded_CircKt = zeros(ComplexF64, 2*Nx,2*Ny,2*Nz)
            #Chi = ifft(FFTCP[cont1, cont2] .* fft(padded_CircKt))
            P = plan_fft(padded_CircKt, flags=FFTW.MEASURE)
            P2Vector[cont1, cont2] = P
            PLI2Vector[cont1, cont2] = plan_ifft(FFTCP[cont1, cont2] .* (P*padded_CircKt), flags=FFTW.MEASURE)
        end
    end
    
    for k = 1:nfreq
        #println("Freq n=$k - Freq Tot=$nfreq")
        # Questa parte la vedremo in futuro (ignoratela per ora tanto QS_Rcc_FW è impostato ad 1)-------------------------------------------------------
        # if QS_Rcc_FW == 2
        #     FFTCLp_rebuilted = compute_Circulant_Lp_Rcc(FFTCLp, escalings, freq[k] ./ escalings["freq"])
        #     FFTCP_rebuilted = compute_Circulant_P_sup_Rcc(FFTCP, escalings, freq[k] ./ escalings["freq"])
        # elseif QS_Rcc_FW == 3
        #     FFTCLp_rebuilted = compute_Circulant_Lp_FW(FFTCLp, escalings, freq[k] ./ escalings["freq"])
        #     FFTCP_rebuilted = compute_Circulant_P_sup_FW(FFTCP, escalings, freq[k] ./ escalings["freq"])
        # end
        # ---------------------------------------------------------------------------

        Yle = build_Yle_S(lumped_elements, [], ports, escalings, n, w[k] / escalings["freq"], R_chiusura)
        Z_self = compute_Z_self(diagonals["R"], diagonals["Cd"], w[k])
        Zs = escalings["R"] * (Zs_info["Zs"] * sqrt(w[k] / escalings["freq"]))
        ind_to_put_zero_Z_self = findall((real.(Zs[Zs_info["surface_edges"]]) .- real.(Z_self[Zs_info["surface_edges"]])) .> 0)
        ind_to_put_zero_Zs = findall((real.(Zs[Zs_info["surface_edges"]]) .- real.(Z_self[Zs_info["surface_edges"]])) .< 0)
        Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]] .= 0 .+ 1im * imag.(Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]])
        Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]] .= 0 .+ 1im * imag.(Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]])
        DZ = Z_self .+ real.(Zs)
        DZ .= DZ .+ 1im * w[k] * diagonals["Lp"]
        invZ = sparse(1:m, 1:m, 1 ./ DZ[:],m,m)
        
        # --------------------- preconditioner ------------------------
        SS = Yle+(prod_real_transposed_complex(incidence_selection["A"] , prod_real_complex(invZ , incidence_selection["A"])) + 1im * w[k] * prod_real_complex(incidence_selection["Gamma"] , invP )* transpose(incidence_selection["Gamma"]))
        F = lu(SS)
        
        # --------------------------------------------------------------
        for c1 = 1:size(ports["port_nodes"], 1)
            n1 = convert(Int64,ports["port_nodes"][c1, 1])
            n2 = convert(Int64,ports["port_nodes"][c1, 2])
            is[n1] = 1 * escalings["Is"]
            is[n2] = -1 * escalings["Is"]
            tn = precond_3_3_Kt(F, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], m, ns, is)
            # products_law = x ->   ComputeMatrixVector(x, w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F, PLIVector, PVector, PLI2Vector, P2Vector,time1Vector, time2Vector)
            # prodts = LinearMap{ComplexF64}(products_law, n + m + ns, n + m + ns)
            # x0 = Vrest[:,c1]
            #prob = LinearProblem(prodts, vec(tn))
        
            if QS_Rcc_FW == 1

                #sol = solve(prob, KrylovJL_GMRES())

                
                V, flag, relres, iter, resvec = gmres_custom(tn, false, GMRES_settings["tol"][k], Inner_Iter, Vrest[:, c1], w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F, PLIVector, PVector, PLI2Vector, P2Vector, time1Vector, time2Vector)
                tot_iter_number = (iter[1] - 1) * Inner_Iter + iter[2] + 1
                if (flag == 0)
                    println("Flag $flag - Iteration = $k - Convergence reached, number of iterations:$tot_iter_number")
                end

                if (flag == 1)
                    println("Flag $flag - Iteration = $k - Convergence not reached, number of iterations:$Inner_Iter")
                end
                # V, info = IterativeSolvers.gmres!(x0, prodts, tn; reltol=GMRES_settings["tol"][k], restart=Inner_Iter, maxiter=Inner_Iter*Inner_Iter, initially_zero=false, log=true, verbose=false)
                # V, info = IterativeSolvers.bicgstabl!(x0, prodts, tn; reltol=GMRES_settings["tol"][k], log=true, max_mv_products=floor(Int, size(prodts,2)/2))
                # V, info = IterativeSolvers.gmres(prodts, tn; reltol=GMRES_settings.tol[k], restart=Inner_Iter, maxiter=Inner_Iter, initially_zero=false, log=true, verbose=false)
                # println(info)
                
                # (V, stats) = Krylov.gmres(prodts,vec(tn), x0, restart=false, memory=Outer_Iter, reorthogonalization=false,rtol=GMRES_settings["tol"][k], itmax=Inner_Iter,verbose=0, history=false, ldiv=true)
                # if (stats.solved==true)        
                #     println("convergence reached, number of iterations: "*string(stats.niter))
                # else
                #     println("convergence not reached, number of iterations: "*string(stats.niter))
                # end
                
                

                # V,flag,err,iter,resvec = KrylovMethods.gmres(prodts,vec(tn),Inner_Iter,tol=GMRES_settings.tol[k],maxIter=1,x=x0,out=1)

                # if (flag==0)        
                #     println("convergence reached, number of iterations: "*string(length(resvec)))
                # elseif(flag==-1)
                #     println("convergence not reached, number of iterations: "*string(length(resvec)))
                # else
                #     println("RHS all zeros, number of iterations: "*string(length(resvec)))
                # end
            else
                #V, flag, relres, iter, resvec = gmres(ComputeMatrixVector, tn, Inner_Iter, GMRES_settings.tol[k], Outer_Iter, [], [], Vrest[:, c1], w[k], incidence_selection, FFTCP_rebuilted, FFTCLp_rebuilted, DZ, Yle, expansions, invZ, invP, L1, U1, P1, Q1)
            end
            
            Vrest[:, c1] = V
            is[n1] = 0
            is[n2] = 0
            for c2 = c1:size(ports["port_nodes"], 1)
                n3 = convert(Int64, ports["port_nodes"][c2, 1])
                n4 = convert(Int64, ports["port_nodes"][c2, 2])
                if c1 == c2
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4]) - R_chiusura) / R_chiusura
                else
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4])) / R_chiusura
                end
                S[c2, c1, k] = S[c1, c2, k]
            end
        end
    end
    println(sum(time1Vector))
    println(sum(time2Vector))
    out::Dict = Dict()
    out["S"] = S
    out["Z"] = s2z(S, R_chiusura)
    out["Y"] = s2y(S, R_chiusura)
    out["f"] = freq ./ escalings["freq"]
    return out
      
end


function precond_3_3_Kt(F, invZ, invP, A, Gamma, n1, n2, X3)
    n3 = length(X3)

    i1 = range(1, stop=n1)
    i2 = range(n1+1, stop=n1 + n2)
    i3 = range(n1 + n2 + 1, stop=n1 + n2 + n3)

    Y = zeros(ComplexF64, n1 + n2 + n3, 1)

    M5 = F\X3
    
    Yi1 = @view Y[i1]
    Y[i1] .= Yi1 .- 1.0*(prod_real_complex(invZ, prod_real_complex(A, M5)))
    Yi2 = @view Y[i2]
    Y[i2] .= Yi2 .+ (prod_real_complex(invP, prod_real_transposed_complex(Gamma, M5)))
    Yi3 = @view Y[i3]
    Y[i3] .= Yi3 .+ M5


    return Y
    
end


function s2z(S,Zo)
    num_ports=size(S)[1]
    nfreq=size(S)[3]
    Z = zeros(ComplexF64 , num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Z[:,:,cont]=Zo*((Id-1.0*S[:,:,cont])\(Id+S[:,:,cont]))
    end
    return Z
end

function s2y(S,Zo)
    num_ports=size(S)[1]
    nfreq=size(S)[3]
    Y = zeros(ComplexF64 , num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Y[:,:,cont]=Zo*((Id+S[:,:,cont])\(Id-1.0*S[:,:,cont]))
    end
    return Y
end