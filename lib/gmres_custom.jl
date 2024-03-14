function gmres_custom(b, restarted, tol, maxit, x , wk, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector)
    m = size(b, 1)
    n = m
    if restarted
        outer = maxit;
        if restart > n
            #warning(message('MATLAB:gmres:tooManyInnerItsRestart',restart, n));
            restart = n;
        end
        inner = restart;
    else
        outer = 1;
        if maxit > n
            #warning(message('MATLAB:gmres:tooManyInnerItsMaxit',maxit, n));
            maxit = n;
        end
        inner = maxit;
    end

    n2b = norm(b);                   
    if (n2b == 0)                    
        a::Union{Vector{ComplexF64}, Vector{Float64}} = zeros(n,1);              
        flag::Int64 = 0;                    
        relres::Float64 = 0.0;                  
        iter::Vector{Int64} = [0 0];                
        resvec::Vector{Float64} = [0.0];                  
        return a, flag, relres, iter, resvec
    end

    if isempty(x)
        x = zeros(ComplexF64,n);
    end

    flag = 1;
    xmin = x;                       
    imin = 0;                        
    jmin = 0;                        
    tolb = tol * n2b;   
    xm = x;             
    evalxm = 0;
    stag = 0;
    moresteps = 0;
    maxmsteps = minimum([floor(n/50),5,n-maxit]);
    maxstagsteps = 3;
    minupdated = 0;
    warned = false;

    r = b - ComputeMatrixVector(x , wk, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector);
    normr = norm(r)
    if normr <= tolb
        x::Union{Vector{ComplexF64}, Vector{Float64}} .= xmin
        flag = 0
        relres = normr / n2b
        iter = [0, 0]
        resvec = [normr]
        return x, flag, relres, iter, resvec
    end
    minv_b = b
    n2minv_b = norm(minv_b)
    tolb = tol * n2minv_b
    
    if normr <= tolb
        x::Union{Vector{ComplexF64}, Vector{Float64}} .= xmin
        flag = 0
        relres = normr / n2minv_b
        iter = [0, 0]
        resvec = [n2minv_b]
        return x, flag, relres, iter, resvec
    end
    resvec = zeros(Float64, inner * outer + 1)
    resvec[1] = normr
    normrmin = normr
    J = zeros(Complex{Float64}, 2, inner)
    U = zeros(Complex{Float64}, n, inner+1)
    R = zeros(Complex{Float64}, inner, inner)
    w = zeros(Complex{Float64}, inner + 1)
    outitercount=0
    for outiter = 1:outer 
        outitercount = outitercount+1
        # Construct u for Householder reflector.
        # u = r + sign(r[1])*norm(r)*e1
        u = copy(r)
        beta = scalarsign(r[1]) * normr
        u[1] += beta
        u /= norm(u)
        
        U[:, 1] = copy(u)
        
        # Apply Householder projection to r.
        # w = r - 2*u*u'*r
        w[1] = -beta
        initercount=0
        for initer = 1:inner
            initercount = initercount + 1
            #println("Iteration = $initercount")
            # Form P1*P2*P3...Pj*ej.
            # v = Pj*ej = ej - 2*u*u'*ej
            
            v = -2 * conj(u[initer]) * u
            v[initer] += 1
            # v = P1*P2*...Pjm1*(Pj*ej)
            for k = initer - 1:-1:1
                @views v -= U[:, k] * (2 * dot(U[:, k], v))
            end
            # Explicitly normalize v to reduce the effects of round-off.
            
            v = v/norm(v)

            
            
            # Apply A to v.
            v = ComputeMatrixVector(v , wk, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector);
            #println(norm(v))
            # Form Pj*Pj-1*...P1*Av.
            for k = 1:initer
                @views v -= U[:, k] * (2 * dot(U[:, k], v))
            end
            
            # Determine Pj+1.
            if initer != length(v)
                # Construct u for Householder reflector Pj+1.
                u = copy(v)
                u[1:initer] .= 0
                alpha = norm(u)
                #println(alpha)
                if alpha != 0
                    alpha = scalarsign(v[initer+1]) * alpha
                    u[initer+1] += alpha
                    u /= norm(u)
                    U[:, initer+1] = copy(u)
                    # Apply Pj+1 to v.
                    v[initer+2:end] .= 0
                    v[initer+1] = -alpha
                end
            end
            
            # Apply Given's rotations to the newly formed v.
            for colJ = 1:initer-1
                tmpv = v[colJ]
                v[colJ] = conj(J[1, colJ]) * v[colJ] + conj(J[2, colJ]) * v[colJ+1]
                v[colJ+1] = -J[2, colJ] * tmpv + J[1, colJ] * v[colJ+1]
            end
            
            # Compute Given's rotation Jm.
            if !(initer == length(v))
                rho = norm(v[initer:initer+1])
                #println(v[initer:initer+1])
                J[:, initer] = v[initer:initer+1] ./ rho
                w[initer+1] = -J[2, initer] .* w[initer]
                w[initer] = conj(J[1, initer]) .* w[initer]
                v[initer] = rho
                v[initer+1] = 0
            end
            
            @views R[:, initer] = v[1:inner]
            normr = abs(w[initer+1])
            resvec[(outiter-1)*inner+initer+1] = normr
            normr_act = normr
            #println(tolb)
            #println(normr)
            if (normr <= tolb || stag >= maxstagsteps || moresteps == 1)
                if evalxm == 0
                    @views ytmp = R[1:initer, 1:initer] \ w[1:initer]
                    @views additive = U[:, initer] * (-2 * ytmp[initer] * conj(U[initer, initer]))
                    additive[initer] += ytmp[initer]
                    for k = initer-1:-1:1
                        additive[k] += ytmp[k]
                        @views additive -= U[:, k] * (2 * dot(U[:, k], additive))
                    end
                    if norm(additive) < eps() * norm(x)
                        stag += 1
                    else
                        stag = 0
                    end
                    xm = x + additive
                    evalxm = 1
                elseif evalxm == 1
                    @views addvc = [-(R[1:initer-1, 1:initer-1] \ R[1:initer-1, initer]) * (w[initer] / R[initer, initer]); w[initer] / R[initer, initer]]
                    if norm(addvc) < eps() * norm(xm)
                        stag += 1
                    else
                        stag = 0
                    end
                    @views additive = U[:, initer] * (-2 * addvc[initer] * conj(U[initer, initer]))
                    additive[initer] += addvc[initer]
                    for k = initer-1:-1:1
                        additive[k] += addvc[k]
                        @views additive -= U[:, k] * (2 * dot(U[:, k], additive))
                    end
                    xm += additive
                end
                r = b - ComputeMatrixVector(xm , wk, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector);
                if norm(r) <= tol * n2b
                    x = xm
                    flag = 0
                    iter = [outiter, initer]
                    break
                end
                minv_r = r
                
                normr_act = norm(minv_r)
                resvec[(outiter-1)*inner+initer+1] = normr_act
                
                if normr_act <= normrmin
                    normrmin = normr_act
                    imin = outiter
                    jmin = initer
                    xmin = xm
                    minupdated = 1
                end
                if normr_act <= tolb
                    x = xm
                    flag = 0
                    iter = [outiter, initer]
                    break
                else
                    if stag >= maxstagsteps && moresteps == 0
                        stag = 0
                    end
                    moresteps += 1
                    if moresteps >= maxmsteps
                        if !warned
                            println("Warning: Tolerance too small, GMRES may not converge.")
                            warned = true
                        end
                        flag = 3
                        iter = [outiter, initer]
                        break
                    end
                end
            end
            
            if normr_act <= normrmin
                normrmin = normr_act
                imin = outiter
                jmin = initer
                minupdated = 1
            end
            
            if stag >= maxstagsteps
                flag = 3
                break
            end
        end
        
        if (initercount == 0)
            initercount = 0
        end
        
        
        evalxm = 0
        
        if flag != 0
            if minupdated == 1
                idx = jmin
            else
                idx = initercount
            end
            if idx > 0 # Allow case inner==0 to flow through
                y = R[1:idx, 1:idx] \ w[1:idx]
                @views additive = U[:, idx] * (-2 * y[idx] * conj(U[idx, idx]))
                additive[idx] += y[idx]
                for k = idx-1:-1:1
                    additive[k] += y[k]
                    @views additive -= U[:, k] * (2 * dot(U[:, k], additive))
                end
                x += additive
            end
            xmin = x
            r = b - ComputeMatrixVector(x , wk, incidence_selection, FFTCP, FFTCLp, DZ, Yle, expansions, invZ, invP, lu, PLIVector, PVector, PLI2Vector, P2Vector, chi2Vector);
            minv_r = r
            normr_act = norm(minv_r)
            r = minv_r
        end
        
        if normr_act <= normrmin
            xmin = x
            normrmin = normr_act
            imin = outiter
            jmin = initercount
        end
        
        if flag == 3
            break
        end
        if normr_act <= tolb
            flag = 0
            iter = [outitercount, initercount]
            break
        end
        minupdated = 0
    end

    if outitercount == 0
        outitercount = 0
        initercount = 0
        normr_act = normrmin
    end
    
    # Returned solution is that with minimum residual
    if flag == 0
        relres = normr_act / n2minv_b
    else
        x = xmin
        iter = [imin, jmin]
        relres = normr_act / n2minv_b
    end
    
    resvec = resvec[1:max(outitercount - 1, 0) * inner + initercount + 1]
    if flag == 2 && initercount != 0
        pop!(resvec)
    end
    
    return x, flag, relres, iter, resvec

end

function scalarsign(d)
    sgn = sign(d)
    if sgn == 0
        sgn = 1
    end
    #println(sgn)
    return sgn
end


