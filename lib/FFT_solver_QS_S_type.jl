using MKL
using SparseArrays, IterativeSolvers, FFTW, LinearAlgebra, LinearMaps
include("build_Yle_S.jl")
include("compute_Z_self.jl")
include("gmres_custom.jl")

function FFT_solver_QS_S_type(freq, escalings, incidence_selection, FFTCP, FFTCLp, diagonals, ports, ports_scatter_value, lumped_elements, expansions, GMRES_settings, Zs_info, QS_Rcc_FW, client, commentsEnabled::Bool)
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
    Vrest = zeros(ComplexF64, m + n + ns, size(ports["port_nodes"], 1))
    invP::SparseMatrixCSC{Float64,Int64} = sparse(1:ns, 1:ns, 1 ./ diagonals["P"], ns, ns)
    PVector::Vector{FFTW.cFFTWPlan{ComplexF64,-1,true,3,Tuple{Int64,Int64,Int64}}} = []
    PLIVector::Vector{AbstractFFTs.ScaledPlan{ComplexF64,FFTW.cFFTWPlan{ComplexF64,1,true,3,UnitRange{Int64}},Float64}} = []
    # ChiVector::Vector{Array{ComplexF64, 3}}=[]
    for cont = 1:3
        Nx::Int64 = size(FFTCLp[cont, 1], 1) ÷ 2
        Ny::Int64 = size(FFTCLp[cont, 1], 2) ÷ 2
        Nz::Int64 = size(FFTCLp[cont, 1], 3) ÷ 2
        padded_CircKt = zeros(ComplexF64, 2 * Nx, 2 * Ny, 2 * Nz)
        push!(PVector, plan_fft!(padded_CircKt, flags=FFTW.MEASURE))
        push!(PLIVector, plan_ifft!(FFTCLp[cont, 1], flags=FFTW.MEASURE))
        # push!(ChiVector, padded_CircKt)
    end

    # P2Vector = Matrix{FFTW.cFFTWPlan{ComplexF64, -1, false, 3, Tuple{Int64, Int64, Int64}}}(undef, 3, 3)
    P2Vector = Matrix{Any}(undef, 3, 3)
    PLI2Vector = Matrix{AbstractFFTs.ScaledPlan{ComplexF64,FFTW.cFFTWPlan{ComplexF64,1,true,3,UnitRange{Int64}},Float64}}(undef, 3, 3)
    Chi2Vector = Matrix{Array{ComplexF64,3}}(undef, 3, 3)
    for cont1 = 1:3
        for cont2 = cont1:3
            Nx::Int64 = size(FFTCP[cont1, cont2], 1) ÷ 2
            Ny::Int64 = size(FFTCP[cont1, cont2], 2) ÷ 2
            Nz::Int64 = size(FFTCP[cont1, cont2], 3) ÷ 2
            padded_CircKt = zeros(ComplexF64, 2 * Nx, 2 * Ny, 2 * Nz)
            #Chi = ifft(FFTCP[cont1, cont2] .* fft(padded_CircKt))
            if cont1 != cont2
                P2Vector[cont1, cont2] = plan_fft(padded_CircKt, flags=FFTW.MEASURE)
                Chi2Vector[cont1, cont2] = padded_CircKt
            else
                P2Vector[cont1, cont2] = plan_fft!(padded_CircKt, flags=FFTW.MEASURE)
            end
            # P2Vector[cont1, cont2] = plan_fft(padded_CircKt, flags=FFTW.MEASURE)
            PLI2Vector[cont1, cont2] = plan_ifft!(FFTCP[cont1, cont2], flags=FFTW.MEASURE)
            # Chi2Vector[cont1, cont2] = padded_CircKt
        end
    end

    resProd = Array{ComplexF64}(undef, 2 * m)
    # resProd = zeros(ComplexF64, 2*m,1)
    # tn = Array{ComplexF64}(undef, m + ns + n)
    tn = zeros(ComplexF64, m + ns + n, 1)

    for k = 1:nfreq
        if length(stopComputation) > 0
            pop!(stopComputation)
            return false
        end
        if !isnothing(client)
            send(client, k)
        end
        Yle::SparseArrays.SparseMatrixCSC{ComplexF64,Int64} = build_Yle_S(lumped_elements, [], ports, escalings, n, w[k] / escalings["freq"], ports_scatter_value)
        Z_self::Vector{ComplexF64} = compute_Z_self(diagonals["R"], diagonals["Cd"], w[k])
        Zs::Matrix{ComplexF64} = escalings["R"] * (Zs_info["Zs"] * sqrt(w[k] / escalings["freq"]))
        Zs_minus_Zself = real.(Zs[Zs_info["surface_edges"]]) .- real.(Z_self[Zs_info["surface_edges"]])
        ind_to_put_zero_Z_self::Vector{Int64} = findall((Zs_minus_Zself) .> 0)
        ind_to_put_zero_Zs::Vector{Int64} = findall((Zs_minus_Zself) .< 0)
        Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]] .= 0 .+ 1im * imag.(Z_self[Zs_info["surface_edges"][ind_to_put_zero_Z_self]])
        Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]] .= 0 .+ 1im * imag.(Zs[Zs_info["surface_edges"][ind_to_put_zero_Zs]])
        DZ::Matrix{ComplexF64} = Z_self .+ real.(Zs)
        DZ .= DZ .+ 1im * w[k] * diagonals["Lp"]
        invZ::SparseArrays.SparseMatrixCSC{ComplexF64,Int64} = sparse(1:m, 1:m, 1 ./ DZ[:], m, m)

        # --------------------- preconditioner ------------------------
        SS::SparseArrays.SparseMatrixCSC{ComplexF64,Int64} = Yle + (transpose(incidence_selection["A"]) * (invZ * incidence_selection["A"])) + 1im * w[k] * (incidence_selection["Gamma"] * invP) * transpose(incidence_selection["Gamma"])
        F::SparseArrays.UMFPACK.UmfpackLU{ComplexF64,Int64} = lu(SS)
        # --------------------------------------------------------------
        for c1::Int64 = 1:size(ports["port_nodes"], 1)
            n1::Int64 = convert(Int32, ports["port_nodes"][c1, 1])
            n2::Int64 = convert(Int32, ports["port_nodes"][c1, 2])
            is[n1] = escalings["Is"]
            # is[n1] = 1 * escalings["Is"]
            is[n2] = -1 * escalings["Is"]
            precond_3_3_Kt!(F, invZ, invP, incidence_selection["A"], incidence_selection["Gamma"], m, ns, vec(is), tn, resProd)
            # products_law = x ->   ComputeMatrixVector(x, w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F, PLIVector, PVector, PLI2Vector, P2Vector,ChiVector, Chi2Vector)
            # prodts = LinearMap{ComplexF64}(products_law, n + m + ns, n + m + ns)
            # x0 = Vrest[:,c1]
            #prob = LinearProblem(prodts, vec(tn))

            if QS_Rcc_FW == 1

                #sol = solve(prob, KrylovJL_GMRES())


                V::Vector{ComplexF64}, flag, relres, iter, resvec = gmres_custom(tn, false, GMRES_settings["tol"][k], Inner_Iter, Vrest[:, c1], w[k], incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, F, PLIVector, PVector, PLI2Vector, P2Vector, Chi2Vector)
                tot_iter_number = (iter[1] - 1) * Inner_Iter + iter[2] + 1
                if commentsEnabled
                    if (flag == 0)
                        println("Flag $flag - Iteration = $k - Convergence reached, number of iterations:$tot_iter_number")
                    end

                    if (flag == 1)
                        println("Flag $flag - Iteration = $k - Convergence not reached, number of iterations:$Inner_Iter")
                    end
                end
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
            for c2::Int64 = c1:size(ports["port_nodes"], 1)
                n3::Int64 = convert(Int32, ports["port_nodes"][c2, 1])
                n4::Int64 = convert(Int32, ports["port_nodes"][c2, 2])
                if c1 == c2
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4]) - ports_scatter_value) / ports_scatter_value
                else
                    S[c1, c2, k] = (2 * (V[m+ns+n3] - V[m+ns+n4])) / ports_scatter_value
                    S[c2, c1, k] = S[c1, c2, k]
                end
            end
        end
    end
    out::Dict = Dict()
    out["S"] = S
    out["Z"] = s2z(S, ports_scatter_value)
    out["Y"] = s2y(S, ports_scatter_value)
    out["f"] = freq ./ escalings["freq"]
    return out

end


function precond_3_3_Kt!(F, invZ, invP, A, Gamma, n1, n2, X3, Y, resProd)
    n3 = length(X3)
    i1 = range(1, stop=n1)
    i2 = range(n1 + 1, stop=n1 + n2)
    i3 = range(n1 + n2 + 1, stop=n1 + n2 + n3)

    # Y = zeros(ComplexF64, n1 + n2 + n3, 1)
    M5 = F \ X3

    # Yi1 = @view Y[i1]
    A_view = @view resProd[1:size(A, 1)]
    invZ_view = @view resProd[size(resProd, 1)-size(invZ, 1)+1:end]
    mul!(A_view, A, M5)
    mul!(invZ_view, invZ, A_view)
    Y[i1] .= lmul!(-1.0, invZ_view)
    # Yi2 = @view Y[i2]
    Gamma_view = @view resProd[size(resProd, 1)-size(Gamma, 2)+1:end]
    mul!(Gamma_view, transpose(Gamma), M5)
    invP_view = @view resProd[1:size(invP, 1)]
    mul!(invP_view, invP, Gamma_view)
    Y[i2] .= invP_view
    # Yi3 = @view Y[i3]
    Y[i3] .= M5

    return Y
end


function s2z(S, Zo)
    num_ports = size(S)[1]
    nfreq = size(S)[3]
    Z = zeros(ComplexF64, num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Z[:, :, cont] = Zo * ((Id - 1.0 * S[:, :, cont]) \ (Id + S[:, :, cont]))
    end
    return Z
end

function s2y(S, Zo)
    num_ports = size(S)[1]
    nfreq = size(S)[3]
    Y = zeros(ComplexF64, num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Y[:, :, cont] = Zo * ((Id + S[:, :, cont]) \ (Id - 1.0 * S[:, :, cont]))
    end
    return Y
end