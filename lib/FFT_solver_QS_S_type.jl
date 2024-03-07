using MKL
using SparseArrays, IterativeSolvers, LinearAlgebra, LinearMaps
include("prod_real_complex.jl")
include("build_Yle_S.jl")
include("compute_Z_self.jl")
include("gmres_custom.jl")

function FFT_solver_QS_S_type(freq, escalings, incidence_selection, FFTCP, FFTCLp, diagonals, ports, lumped_elements, expansions, GMRES_settings, Zs_info, QS_Rcc_FW)
    #FFTW.set_num_threads(Threads.nthreads())
    # FFTW.set_num_threads(12)
    #BLAS.set_num_threads(convert(Int64,Base.Threads.nthreads()/2))
    #BLAS.set_num_threads(6)
    
    freq = freq .* escalings["freq"]
    # GMRES settings ----------------------------
    
    Inner_Iter::Int64 = GMRES_settings["Inner_Iter"]
    #Outer_Iter = GMRES_settings.Outer_Iter
    # -------------------------------------------
    m::Int64 = size(incidence_selection["A"], 1)
    n::Int64 = size(incidence_selection["A"], 2)
    ns::Int64 = size(incidence_selection["Gamma"], 2)
    w = 2 .* pi .* freq
    nfreq = length(w)
    is = zeros(Float64, n, 1)
    S = zeros(ComplexF64, size(ports["port_nodes"], 1), size(ports["port_nodes"], 1), length(freq))
    Vrest::Matrix{ComplexF64} = zeros(ComplexF64, m + n + ns, size(ports["port_nodes"], 1))
    invP::SparseMatrixCSC{Float64, Int64} = sparse(1:ns, 1:ns, 1 ./ diagonals["P"],ns,ns)
    R_chiusura = 50.0
    
    for k = 1:nfreq
        Yle::SparseArrays.SparseMatrixCSC{Float64, Int64} = build_Yle_S(lumped_elements, [], ports, escalings, n, w[k] / escalings["freq"], R_chiusura)
        Z_self::Vector{ComplexF64} = compute_Z_self(diagonals["R"], diagonals["Cd"], w[k])
        Zs::Matrix{ComplexF64} = escalings["R"] * (Zs_info["Zs"] * sqrt(w[k] / escalings["freq"]))
        ind_to_put_zero_Z_self::Vector{Int64} = findall((real.(Zs[Zs_info["surface_edges"]]) .- real.(Z_self[Zs_info["surface_edges"]])) .> 0)
        ind_to_put_zero_Zs::Vector{Int64} = findall((real.(Zs[Zs_info["surface_edges"]]) .- real.(Z_self[Zs_info["surface_edges"]])) .< 0)
        Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]] .= 0 .+ 1im * imag.(Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]])
        Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]] .= 0 .+ 1im * imag.(Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]])
        DZ::Matrix{ComplexF64} = Z_self .+ real.(Zs)
        DZ .= DZ .+ 1im * w[k] * diagonals["Lp"]
        invZ::SparseArrays.SparseMatrixCSC{ComplexF64, Int64} = sparse(1:m, 1:m, 1 ./ DZ[:],m,m)
        
        # --------------------- preconditioner ------------------------
        SS::SparseArrays.SparseMatrixCSC{ComplexF64, Int64} = Yle+(prod_real_transposed_complex(incidence_selection["A"] , prod_real_complex(invZ , incidence_selection["A"])) + 1im * w[k] * prod_real_complex(incidence_selection["Gamma"] , invP )* transpose(incidence_selection["Gamma"]))
        F::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64} = lu(SS)
        # --------------------------------------------------------------
        for c1::Int64 = 1:size(ports["port_nodes"], 1)
            n1::Int64 = convert(Int32,ports["port_nodes"][c1, 1])
            n2::Int64 = convert(Int32,ports["port_nodes"][c1, 2])
            is[n1] = 1 * escalings["Is"]
            is[n2] = -1 * escalings["Is"]
            tn = precond_3_3_Kt(F, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], m, ns, is)
            # products_law = x ->   ComputeMatrixVector(x, w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F, PLIVector, PVector, PLI2Vector, P2Vector,ChiVector, Chi2Vector)
            # prodts = LinearMap{ComplexF64}(products_law, n + m + ns, n + m + ns)
            # x0 = Vrest[:,c1]
            #prob = LinearProblem(prodts, vec(tn))
        
            if QS_Rcc_FW == 1

                #sol = solve(prob, KrylovJL_GMRES())

                
                V::Vector{ComplexF64}, flag, relres, iter, resvec = gmres_custom(tn, false, GMRES_settings["tol"][k], Inner_Iter, Vrest[:, c1], w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F)
                tot_iter_number = (iter[1] - 1) * Inner_Iter + iter[2] + 1
                if (flag == 0)
                    println("Flag $flag - Iteration = $k - Convergence reached, number of iterations:$tot_iter_number")
                end

                # if (flag == 1)
                #     println("Flag $flag - Iteration = $k - Convergence not reached, number of iterations:$Inner_Iter")
                # end
                #V, info = IterativeSolvers.gmres!(x0, prodts, tn; reltol=GMRES_settings.tol[k], restart=Inner_Iter, maxiter=Inner_Iter, initially_zero=false, log=true, verbose=false)
                # V, info = IterativeSolvers.gmres(prodts, tn; reltol=GMRES_settings.tol[k], restart=Inner_Iter, maxiter=Inner_Iter, initially_zero=false, log=true, verbose=false)
                #println(info)
                
                # (V, stats) = Krylov.gmres(prodts,vec(tn), x0;restart=false, memory=Outer_Iter, reorthogonalization=false,rtol=GMRES_settings.tol[k], itmax=Inner_Iter,verbose=0, history=false, ldiv=true)
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
                n3 = convert(Int32, ports["port_nodes"][c2, 1])
                n4 = convert(Int32, ports["port_nodes"][c2, 2])
                if c1 == c2
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4]) - R_chiusura) / R_chiusura
                else
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4])) / R_chiusura
                end
                S[c2, c1, k] = S[c1, c2, k]
            end
        end
    end
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
    Y[i1] .= Yi1 .- lmul!(1.0, prod_real_complex(invZ, prod_real_complex(A, M5)))
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