function compute_Zs_with_indices(materials, li_mats, maps_Zs, sx, sy, sz)
    mx = size(li_mats["lix_mat"], 1)
    my = size(li_mats["liy_mat"], 1)
    mz = size(li_mats["liz_mat"], 1)
    contenitore = zeros(length(materials), 6) .+ 1im .* zeros(length(materials), 6)
    cont_cond = 0
    cont_diel = 0
    for cont in range(1,length(materials))
        eps_re = materials[cont]["eps_re"]
        if eps_re > 1
            cont_diel = cont_diel + 1
        end
    end
    id_diel = zeros(cont_diel, 1)
    cont_diel = 0
    for cont in range(1,length(materials))
        eps_re = materials[cont]["eps_re"]
        if eps_re > 1
            cont_diel = cont_diel + 1
            id_diel[cont_diel] = cont
        end
    end
    for cont in range(1,length(materials))
        sigmar = materials[cont]["sigmar"]
        if sigmar != 0
            cont_cond = cont_cond + 1
            contenitore[cont, 1] = 0.5 * sx / sy * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
            contenitore[cont, 2] = 0.5 * sx / sz * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
            contenitore[cont, 3] = 0.5 * sy / sx * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
            contenitore[cont, 4] = 0.5 * sy / sz * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
            contenitore[cont, 5] = 0.5 * sz / sx * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
            contenitore[cont, 6] = 0.5 * sz / sy * (1 + 1im) * sqrt(2 * pi * 1e-7 / sigmar)
        end
    end
    Ntot = cont_cond
    Z_surf_x = Array{Any}(undef,4,Ntot)
    Z_surf_y = Array{Any}(undef,4,Ntot)
    Z_surf_z = Array{Any}(undef,4,Ntot)
    Z_surf_rug_x = Array{Any}(undef,4,Ntot)
    Z_surf_rug_y = Array{Any}(undef,4,Ntot)
    Z_surf_rug_z = Array{Any}(undef,4,Ntot)
    ind_cond_x = Array{Any}(undef,4,Ntot)
    ind_cond_y = Array{Any}(undef,4,Ntot)
    ind_cond_z = Array{Any}(undef,4,Ntot)
    ind_cond_rug_x = Array{Any}(undef,4,Ntot)
    ind_cond_rug_y = Array{Any}(undef,4,Ntot)
    ind_cond_rug_z = Array{Any}(undef,4,Ntot)
    for k1 = 1:4
        for k2 = 1:Ntot
            Z_surf_x[k1, k2] = []
            Z_surf_y[k1, k2] = []
            Z_surf_z[k1, k2] = []
            Z_surf_rug_x[k1, k2] = []
            Z_surf_rug_y[k1, k2] = []
            Z_surf_rug_z[k1, k2] = []
            ind_cond_x[k1, k2] = []
            ind_cond_y[k1, k2] = []
            ind_cond_z[k1, k2] = []
            ind_cond_rug_x[k1, k2] = []
            ind_cond_rug_y[k1, k2] = []
            ind_cond_rug_z[k1, k2] = []
        end
    end
    Zswx = zeros(mx, 4) .+ 1im .* zeros(mx, 4)
    Zswy = zeros(my, 4) .+ 1im .* zeros(my, 4)
    Zswz = zeros(mz, 4) .+ 1im .* zeros(mz, 4)
    ind_x_xy_1 = findall(maps_Zs["x_xy"][:, 1] .> 0)
    ind_x_xy_2 = findall(maps_Zs["x_xy"][:, 2] .> 0)
    ind_x_zx_1 = findall(maps_Zs["x_zx"][:, 1] .> 0)
    ind_x_zx_2 = findall(maps_Zs["x_zx"][:, 2] .> 0)
    ind_y_xy_1 = findall(maps_Zs["y_xy"][:, 1] .> 0)
    ind_y_xy_2 = findall(maps_Zs["y_xy"][:, 2] .> 0)
    ind_y_yz_1 = findall(maps_Zs["y_yz"][:, 1] .> 0)
    ind_y_yz_2 = findall(maps_Zs["y_yz"][:, 2] .> 0)
    ind_z_zx_1 = findall(maps_Zs["z_zx"][:, 1] .> 0)
    ind_z_zx_2 = findall(maps_Zs["z_zx"][:, 2] .> 0)
    ind_z_yz_1 = findall(maps_Zs["z_yz"][:, 1] .> 0)
    ind_z_yz_2 = findall(maps_Zs["z_yz"][:, 2] .> 0)
    cont_cond = 0
    for cont in range(1,length(materials))
        if contenitore[cont, 1] != 0
            cont_cond = cont_cond + 1
            ind_m = findall(cont .== li_mats["lix_mat"][:, 1])
            ind_coder_1 = intersect(ind_x_xy_1, ind_m)
            ind_cond_x[1, cont_cond] = ind_coder_1
            ind_rug_1 = zeros(mx, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_x_xy_cd = findall(maps_Zs["x_xy"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_1, ind_x_xy_cd)
                        ind_rug_1[ind_rg] .= 1
                    end
                end
            end
            ind_rug_1 = findall(x -> x > 0, vec(ind_rug_1))
            ind_cond_rug_x[1, cont_cond] = ind_rug_1
            Zswx[ind_coder_1, 1] .= contenitore[cont, 1]
            ind_coder_3 = intersect(ind_x_xy_2, ind_m)
            ind_cond_x[3, cont_cond] = ind_coder_3
            ind_rug_3 = zeros(mx, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_x_xy_cd = findall(maps_Zs["x_xy"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_3, ind_x_xy_cd)
                        ind_rug_3[ind_rg] .= 1
                    end
                end
            end
            ind_rug_3 = findall(x -> x > 0, vec(ind_rug_3))
            ind_cond_rug_x[3, cont_cond] = ind_rug_3
            Zswx[ind_coder_3, 3] .= contenitore[cont, 1]
            ind_coder_2 = intersect(ind_x_zx_1, ind_m)
            ind_cond_x[2, cont_cond] = ind_coder_2
            ind_rug_2 = zeros(mx, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_x_zx_cd = findall(maps_Zs["x_zx"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_2, ind_x_zx_cd)
                        ind_rug_2[ind_rg] .= 1
                    end
                end
            end
            ind_rug_2 = findall(x -> x > 0, vec(ind_rug_2))
            ind_cond_rug_x[2, cont_cond] = ind_rug_2
            Zswx[ind_coder_2, 2] .= contenitore[cont, 2]
            ind_coder_4 = intersect(ind_x_zx_2, ind_m)
            ind_cond_x[4, cont_cond] = ind_coder_4
            ind_rug_4 = zeros(mx, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_x_zx_cd = findall(maps_Zs["x_zx"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_4, ind_x_zx_cd)
                        ind_rug_4[ind_rg] .= 1
                    end
                end
            end
            ind_rug_4 = findall(x -> x > 0, vec(ind_rug_4))
            ind_cond_rug_x[4, cont_cond] = ind_rug_4
            Zswx[ind_coder_4, 4] .= contenitore[cont, 2]
            ind_m = findall(cont .== li_mats["lix_border"][:, 1])
            ind = intersect(ind_coder_1, ind_m)
            Zswx[ind, 1] = 2 * Zswx[ind, 1]
            ind = intersect(ind_coder_2, ind_m)
            Zswx[ind, 2] = 2 * Zswx[ind, 2]
            Z_surf_x[1, cont_cond] = Zswx[ind_coder_1, 1]
            Z_surf_x[2, cont_cond] = Zswx[ind_coder_2, 2]
            Z_surf_rug_x[1, cont_cond] = Zswx[ind_rug_1]
            Z_surf_rug_x[2, cont_cond] = Zswx[ind_rug_2]
            ind_m = findall(cont .== li_mats["lix_border"][:, 2])
            ind = intersect(ind_coder_3, ind_m)
            Zswx[ind, 3] = 2 * Zswx[ind, 3]
            ind = intersect(ind_coder_4, ind_m)
            Zswx[ind, 4] = 2 * Zswx[ind, 4]
            Z_surf_x[3, cont_cond] = Zswx[ind_coder_3, 3]
            Z_surf_x[4, cont_cond] = Zswx[ind_coder_4, 4]
            Z_surf_rug_x[3, cont_cond] = Zswx[ind_rug_3]
            Z_surf_rug_x[4, cont_cond] = Zswx[ind_rug_4]
            ind_m = findall(cont .== li_mats["liy_mat"][:, 1])
            ind_coder_1 = intersect(ind_y_xy_1, ind_m)
            ind_cond_y[1, cont_cond] = ind_coder_1
            ind_rug = zeros(my, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_y_xy_cd = findall(maps_Zs["y_xy"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_1, ind_y_xy_cd)
                        ind_rug[ind_rg] .= 1
                    end
                end
            end
            ind_rug_1 = findall(x -> x > 0, vec(ind_rug))
            ind_cond_rug_y[1, cont_cond] = ind_rug_1
            Zswy[ind_coder_1, 1] .= contenitore[cont, 3]
            ind_coder_3 = intersect(ind_y_xy_2, ind_m)
            ind_cond_y[3, cont_cond] = ind_coder_3
            ind_rug_3 = zeros(my, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_y_xy_cd = findall(maps_Zs["y_xy"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_3, ind_y_xy_cd)
                        ind_rug_3[ind_rg] .= 1
                    end
                end
            end
            ind_rug_3 = findall(x -> x > 0, vec(ind_rug_3))
            ind_cond_rug_y[3, cont_cond] = ind_rug_3
            Zswy[ind_coder_3, 3] .= contenitore[cont, 3]
            ind_coder_2 = intersect(ind_y_yz_1, ind_m)
            ind_cond_y[2, cont_cond] = ind_coder_2
            ind_rug_2 = zeros(my, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_y_yz_cd = findall(maps_Zs["y_yz"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_2, ind_y_yz_cd)
                        ind_rug_2[ind_rg] .= 1
                    end
                end
            end
            ind_rug_2 = findall(x -> x > 0, vec(ind_rug_2))
            ind_cond_rug_y[2, cont_cond] = ind_rug_2
            Zswy[ind_coder_2, 2] .= contenitore[cont, 4]
            ind_coder_4 = intersect(ind_y_yz_2, ind_m)
            ind_cond_y[4, cont_cond] = ind_coder_4
            ind_rug_4 = zeros(my, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_y_yz_cd = findall(maps_Zs["y_yz"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_4, ind_y_yz_cd)
                        ind_rug_4[ind_rg] .= 1
                    end
                end
            end
            ind_rug_4 = findall(x -> x > 0, vec(ind_rug_4))
            ind_cond_rug_y[4, cont_cond] = ind_rug_4
            Zswy[ind_coder_4, 4] .= contenitore[cont, 4]
            ind_m = findall(cont .== li_mats["liy_border"][:, 1])
            ind = intersect(ind_coder_1, ind_m)
            Zswy[ind, 1] = 2 * Zswy[ind, 1]
            ind = intersect(ind_coder_2, ind_m)
            Zswy[ind, 2] = 2 * Zswy[ind, 2]
            Z_surf_y[1, cont_cond] = Zswy[ind_coder_1, 1]
            Z_surf_y[2, cont_cond] = Zswy[ind_coder_2, 2]
            Z_surf_rug_y[1, cont_cond] = Zswy[ind_rug_1]
            Z_surf_rug_y[2, cont_cond] = Zswy[ind_rug_2]
            ind_m = findall(cont .== li_mats["liy_border"][:, 2])
            ind = intersect(ind_coder_3, ind_m)
            Zswy[ind, 3] = 2 * Zswy[ind, 3]
            ind = intersect(ind_coder_4, ind_m)
            Zswy[ind, 4] = 2 * Zswy[ind, 4]
            Z_surf_y[3, cont_cond] = Zswy[ind_coder_3, 3]
            Z_surf_y[4, cont_cond] = Zswy[ind_coder_4, 4]
            Z_surf_rug_y[3, cont_cond] = Zswy[ind_rug_3]
            Z_surf_rug_y[4, cont_cond] = Zswy[ind_rug_4]
            ind_m = findall(cont .== li_mats["liz_mat"][:, 1])
            ind_coder_1 = intersect(ind_z_zx_1, ind_m)
            ind_cond_z[1, cont_cond] = ind_coder_1
            ind_rug_1 = zeros(mz, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_z_zx_cd = findall(maps_Zs["z_zx"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_1, ind_z_zx_cd)
                        ind_rug_1[ind_rg] .= 1
                    end
                end
            end
            ind_rug_1 = findall(x -> x > 0, vec(ind_rug_1))
            ind_cond_rug_z[1, cont_cond] = ind_rug_1
            Zswz[ind_coder_1, 1] .= contenitore[cont, 5]
            ind_coder_3 = intersect(ind_z_zx_2, ind_m)
            ind_cond_z[3, cont_cond] = ind_coder_3
            ind_rug_3 = zeros(mz, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_z_zx_cd = findall(maps_Zs["z_zx"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_3, ind_z_zx_cd)
                        ind_rug_3[ind_rg] .= 1
                    end
                end
            end
            ind_rug_3 = findall(x -> x > 0, vec(ind_rug_3))
            ind_cond_rug_z[3, cont_cond] = ind_rug_3
            Zswz[ind_coder_3, 3] .= contenitore[cont, 5]
            ind_coder_2 = intersect(ind_z_yz_1, ind_m)
            ind_cond_z[2, cont_cond] = ind_coder_2
            ind_rug_2 = zeros(mz, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_z_yz_cd = findall(maps_Zs["z_yz"][:, 3] .== cont2)
                        ind_rg = intersect(ind_coder_2, ind_z_yz_cd)
                        ind_rug_2[ind_rg] .= 1
                    end
                end
            end
            ind_rug_2 = findall(x -> x > 0, vec(ind_rug_2))
            ind_cond_rug_z[2, cont_cond] = ind_rug_2
            Zswz[ind_coder_2, 2] .= contenitore[cont, 6]
            ind_coder_4 = intersect(ind_z_yz_2, ind_m)
            ind_cond_z[4, cont_cond] = ind_coder_4
            ind_rug_4 = zeros(mz, 1)
            for cont2 = 1:length(materials)
                if cont != cont2
                    if !isempty(findall(!iszero, id_diel)) == false
                        ind_z_yz_cd = findall(maps_Zs["z_yz"][:, 4] .== cont2)
                        ind_rg = intersect(ind_coder_4, ind_z_yz_cd)
                        ind_rug_4[ind_rg] .= 1
                    end
                end
            end
            ind_rug_4 = findall(x -> x > 0, vec(ind_rug_4))
            ind_cond_rug_z[4, cont_cond] = ind_rug_4
            Zswz[ind_coder_4, 4] .= contenitore[cont, 6]
            ind_m = findall(cont .== li_mats["liz_border"][:, 1])
            ind = intersect(ind_coder_1, ind_m)
            Zswz[ind, 1] = 2 * Zswz[ind, 1]
            ind = intersect(ind_coder_2, ind_m)
            Zswz[ind, 2] = 2 * Zswz[ind, 2]
            Z_surf_z[1, cont_cond] = Zswz[ind_coder_1, 1]
            Z_surf_z[2, cont_cond] = Zswz[ind_coder_2, 2]
            Z_surf_rug_z[1, cont_cond] = Zswz[ind_rug_1]
            Z_surf_rug_z[2, cont_cond] = Zswz[ind_rug_2]
            ind_m = findall(cont .== li_mats["liz_border"][:, 2])
            ind = intersect(ind_coder_3, ind_m)
            Zswz[ind, 3] = 2 * Zswz[ind, 3]
            ind = intersect(ind_coder_4, ind_m)
            Zswz[ind, 4] = 2 * Zswz[ind, 4]
            Z_surf_z[3, cont_cond] = Zswz[ind_coder_3, 3]
            Z_surf_z[4, cont_cond] = Zswz[ind_coder_4, 4]
            Z_surf_rug_z[3, cont_cond] = Zswz[ind_rug_3]
            Z_surf_rug_z[4, cont_cond] = Zswz[ind_rug_4]
        end
    end
    z1_x = zeros(mx, 1) .+ 1im .* zeros(mx, 1)
    z2_x = zeros(mx, 1) .+ 1im .* zeros(mx, 1)
    z3_x = zeros(mx, 1) .+ 1im .* zeros(mx, 1)
    z4_x = zeros(mx, 1) .+ 1im .* zeros(mx, 1)
    k = 0
    for cont in range(1,length(materials))
        sigmar = materials[cont]["sigmar"]
        if sigmar != 0
            k = k + 1
            z1_x[ind_cond_x[1, k]] = Z_surf_x[1, k]
            z2_x[ind_cond_x[2, k]] = Z_surf_x[2, k]
            z3_x[ind_cond_x[3, k]] = Z_surf_x[3, k]
            z4_x[ind_cond_x[4, k]] = Z_surf_x[4, k]
        end
    end
    ind1 = findall(x -> abs(x) != 0, vec(z1_x))
    ind2 = findall(x -> abs(x) != 0, vec(z2_x))
    ipara = intersect(ind1, ind2)
    zsx1 = z1_x + z2_x
    zsx1[ipara] = z1_x[ipara] .* z2_x[ipara] ./ (z1_x[ipara] + z2_x[ipara])
    ind1 = findall(x -> abs(x) != 0, vec(z3_x))
    ind2 = findall(x -> abs(x) != 0, vec(z4_x))
    ipara = intersect(ind1, ind2)
    zsx2 = z3_x + z4_x
    zsx2[ipara] = z3_x[ipara] .* z4_x[ipara] ./ (z3_x[ipara] + z4_x[ipara])
    zsx = zsx1 + zsx2
    z1_y = zeros(my, 1) .+ 1im .* zeros(my, 1)
    z2_y = zeros(my, 1) .+ 1im .* zeros(my, 1)
    z3_y = zeros(my, 1) .+ 1im .* zeros(my, 1)
    z4_y = zeros(my, 1) .+ 1im .* zeros(my, 1)
    k = 0
    for cont in range(1,length(materials))
        sigmar = materials[cont]["sigmar"]
        if sigmar != 0
            k = k + 1
            z1_y[ind_cond_y[1, k]] = Z_surf_y[1, k]
            z2_y[ind_cond_y[2, k]] = Z_surf_y[2, k]
            z3_y[ind_cond_y[3, k]] = Z_surf_y[3, k]
            z4_y[ind_cond_y[4, k]] = Z_surf_y[4, k]
        end
    end
    ind1 = findall(x -> abs(x) != 0, vec(z1_y))
    ind2 = findall(x -> abs(x) != 0, vec(z2_y))
    ipara = intersect(ind1, ind2)
    zsy1 = z1_y + z2_y
    zsy1[ipara] = z1_y[ipara] .* z2_y[ipara] ./ (z1_y[ipara] + z2_y[ipara])
    ind1 = findall(x -> abs(x) != 0, vec(z3_y))
    ind2 = findall(x -> abs(x) != 0, vec(z4_y))
    ipara = intersect(ind1, ind2)
    zsy2 = z3_y + z4_y
    zsy2[ipara] = z3_y[ipara] .* z4_y[ipara] ./ (z3_y[ipara] + z4_y[ipara])
    zsy = zsy1 + zsy2
    z1_z = zeros(mz, 1) .+ 1im .* zeros(mz, 1)
    z2_z = zeros(mz, 1) .+ 1im .* zeros(mz, 1)
    z3_z = zeros(mz, 1) .+ 1im .* zeros(mz, 1)
    z4_z = zeros(mz, 1) .+ 1im .* zeros(mz, 1)
    k = 0
    for cont in range(1,length(materials))
        sigmar = materials[cont]["sigmar"]
        if sigmar != 0
            k = k + 1
            z1_z[ind_cond_z[1, k]] = Z_surf_z[1, k]
            z2_z[ind_cond_z[2, k]] = Z_surf_z[2, k]
            z3_z[ind_cond_z[3, k]] = Z_surf_z[3, k]
            z4_z[ind_cond_z[4, k]] = Z_surf_z[4, k]
        end
    end
    ind1 = findall(x -> abs(x) != 0, vec(z1_z))
    ind2 = findall(x -> abs(x) != 0, vec(z2_z))
    ipara = intersect(ind1, ind2)
    zsz1 = z1_z + z2_z
    zsz1[ipara] = z1_z[ipara] .* z2_z[ipara] ./ (z1_z[ipara] + z2_z[ipara])
    ind1 = findall(x -> abs(x) != 0, vec(z3_z))
    ind2 = findall(x -> abs(x) != 0, vec(z4_z))
    ipara = intersect(ind1, ind2)
    zsz2 = z3_z + z4_z
    zsz2[ipara] = z3_z[ipara] .* z4_z[ipara] ./ (z3_z[ipara] + z4_z[ipara])
    zsz = zsz1 + zsz2
    Zs_info_Zs = [zsx; zsy; zsz]
    Zs_info_internal_edges = findall(x -> abs(x) == 0 ,vec(Zs_info_Zs))
    Zs_info_surface_edges = setdiff(1:mx + my + mz, Zs_info_internal_edges)
    Zrg_x = [z1_x z2_x z3_x z4_x]
    Zrg_y = [z1_y z2_y z3_y z4_y]
    Zrg_z = [z1_z z2_z z3_z z4_z]
    Zs_info_materials = [li_mats["lix_mat"][:, 1]; li_mats["liy_mat"][:, 1]; li_mats["liz_mat"][:, 1]]
    Zs_info = Dict("Zs" => complex(Zs_info_Zs), "internal_edges" => Zs_info_internal_edges,
        "surface_edges" => Zs_info_surface_edges, "Zrg_x" => complex(Zrg_x),
        "Zrg_y" => complex(Zrg_y), "Zrg_z" => complex(Zrg_z),
        "materials" => Zs_info_materials)
    return Zs_info,ind_cond_rug_x,ind_cond_rug_y,ind_cond_rug_z
end