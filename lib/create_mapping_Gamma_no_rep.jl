include("From_3D_to_1D.jl")
include("bin_search.jl")
using SparseArrays

function create_mapping_Gamma_no_rep(grids, map_volumes, nodes, nodes_red, externals_grids)
    Nx = size(grids[1], 1)
    Ny = size(grids[1][1], 1)
    Nz = size(grids[1][1][1], 1)
    num_grids = length(grids)
    mapping_surf_12_se = zeros((Nx) * (Ny + 1) * (Nz))
    mapping_surf_34_se = zeros((Nx + 1) * (Ny) * (Nz))
    mapping_surf_56_se = zeros((Nx) * (Ny) * (Nz + 1))
    mapping_surf_1234 = zeros((2 * (Nx + 1) - 1) * (2 * (Ny + 1) - 1) * (Nz))
    mapping_surf_1256 = zeros((Nx) * (2 * (Ny + 1) - 1) * (2 * (Nz + 1) - 1))
    mapping_surf_3456 = zeros((2 * (Nx + 1) - 1) * (Ny) * (2 * (Nz + 1) - 1))
    num_ele_12 = 0
    num_ele_34 = 0
    num_ele_56 = 0
    nnz_surf_max = 3 * Nx * Ny * Nz
    ind_r = zeros(nnz_surf_max)
    ind_c = zeros(nnz_surf_max)
    contat_tot = 0
    for cont = 1:Nx
        cont2 = 1
        # for cont2 = 1
        for cont3 = 1:Nz
            for k = 1:num_grids
                check_om = false
                for k2 = 1:num_grids
                    if k != k2 && cont2 + 1 <= Ny
                        if grids[k2][cont][cont2+1][cont3]
                            check_om = true
                        end
                    end
                end
                if grids[k][cont][cont2][cont3] && (!externals_grids[k, 2][cont, cont2, cont3] || check_om)
                    num_ele_12 += 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_12_se[From_3D_to_1D(cont, cont2, cont3, Nx, Ny + 1)] = num_ele_12
                    mapping_surf_1234[From_3D_to_1D(cont * 2, cont2 * 2 - 1, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12
                    mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2 - 1, cont3 * 2, Nx, 2 * (Ny + 1) - 1)] = num_ele_12
                    contat_tot += 1
                    ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                    ind_c[contat_tot] = num_ele_12
                    break
                end
            end
        end
        # end
    end
    for cont = 1:Nx
        for cont2 = 2:Ny
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont2 + 1 <= Ny
                            if grids[k2][cont][cont2+1][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !grids[k][cont][cont2-1][cont3] && (!externals_grids[k, 2][cont, cont2, cont3] || check_om)
                        num_ele_12 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_12_se[From_3D_to_1D(cont, cont2, cont3, Nx, Ny + 1)] = num_ele_12
                        mapping_surf_1234[From_3D_to_1D(cont * 2, cont2 * 2 - 1, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12
                        mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2 - 1, cont3 * 2, Nx, 2 * (Ny + 1) - 1)] = num_ele_12
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx
        for cont2 = Ny
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont2 - 1 >= 1
                            if grids[k2][cont][cont2-1][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && (!externals_grids[k, 1][cont, cont2, cont3] || check_om)
                        num_ele_12 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_12_se[From_3D_to_1D(cont, cont2 + 1, cont3, Nx, Ny + 1)] = num_ele_12
                        mapping_surf_1234[From_3D_to_1D(cont * 2, (cont2 + 1) * 2 - 1, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12
                        mapping_surf_1256[From_3D_to_1D(cont, (cont2 + 1) * 2 - 1, cont3 * 2, Nx, 2 * (Ny + 1) - 1)] = num_ele_12
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx
        for cont2 = 1:Ny-1
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont2 - 1 >= 1
                            if grids[k2][cont][cont2-1][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !grids[k][cont][cont2+1][cont3] && (!externals_grids[k, 1][cont, cont2, cont3] || check_om)
                        controllo_altri = 0
                        for k2 = 1:num_grids
                            if k != k2
                                if grids[k2][cont][cont2+1][cont3]
                                    controllo_altri = 1
                                    break
                                end
                            end
                        end
                        if controllo_altri == 0
                            num_ele_12 += 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_12_se[From_3D_to_1D(cont, cont2 + 1, cont3, Nx, Ny + 1)] = num_ele_12
                            mapping_surf_1234[From_3D_to_1D(cont * 2, (cont2 + 1) * 2 - 1, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12
                            mapping_surf_1256[From_3D_to_1D(cont, (cont2 + 1) * 2 - 1, cont3 * 2, Nx, 2 * (Ny + 1) - 1)] = num_ele_12
                            contat_tot += 1
                            ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                            ind_c[contat_tot] = num_ele_12
                        end
                        break
                    end
                end
            end
        end
    end
    cont = 1
    # for cont = 1
    for cont2 = 1:Ny
        for cont3 = 1:Nz
            for k = 1:num_grids
                check_om = false
                for k2 = 1:num_grids
                    if k != k2 && cont + 1 <= Nx
                        if grids[k2][cont+1][cont2][cont3]
                            check_om = true
                        end
                    end
                end
                if grids[k][cont][cont2][cont3] && (!externals_grids[k, 4][cont, cont2, cont3] || check_om)
                    num_ele_34 += 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_34_se[From_3D_to_1D(cont, cont2, cont3, Nx + 1, Ny)] = num_ele_12 + num_ele_34
                    mapping_surf_1234[From_3D_to_1D(cont * 2 - 1, cont2 * 2, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34
                    mapping_surf_3456[From_3D_to_1D(cont * 2 - 1, cont2, cont3 * 2, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34
                    contat_tot += 1
                    ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                    ind_c[contat_tot] = num_ele_12 + num_ele_34
                    break
                end
            end
        end
    end
    # end
    for cont = 2:Nx
        for cont2 = 1:Ny
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont + 1 <= Nx
                            if grids[k2][cont+1][cont2][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !(grids[k][cont-1][cont2][cont3]) && (!externals_grids[k, 4][cont, cont2, cont3] || check_om)
                        num_ele_34 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_34_se[From_3D_to_1D(cont, cont2, cont3, Nx + 1, Ny)] = num_ele_12 + num_ele_34
                        mapping_surf_1234[From_3D_to_1D(cont * 2 - 1, cont2 * 2, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34
                        mapping_surf_3456[From_3D_to_1D(cont * 2 - 1, cont2, cont3 * 2, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12 + num_ele_34
                        break
                    end
                end
            end
        end
    end
    for cont = Nx
        for cont2 = 1:Ny
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont - 1 >= 1
                            if grids[k2][cont-1][cont2][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && (!(externals_grids[k, 3][cont, cont2, cont3]) || check_om)
                        num_ele_34 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_34_se[From_3D_to_1D(cont + 1, cont2, cont3, Nx + 1, Ny)] = num_ele_12 + num_ele_34
                        mapping_surf_1234[From_3D_to_1D((cont + 1) * 2 - 1, cont2 * 2, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34
                        mapping_surf_3456[From_3D_to_1D((cont + 1) * 2 - 1, cont2, cont3 * 2, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12 + num_ele_34
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx-1
        for cont2 = 1:Ny
            for cont3 = 1:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont - 1 >= 1
                            if grids[k2][cont-1][cont2][cont3]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !(grids[k][cont+1][cont2][cont3]) && (!externals_grids[k, 3][cont, cont2, cont3] || check_om)
                        controllo_altri = 0
                        for k2 = 1:num_grids
                            if k != k2
                                if grids[k2][cont+1][cont2][cont3]
                                    controllo_altri = 1
                                    break
                                end
                            end
                        end
                        if controllo_altri == 0
                            num_ele_34 += 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_34_se[From_3D_to_1D(cont + 1, cont2, cont3, Nx + 1, Ny)] = num_ele_12 + num_ele_34
                            mapping_surf_1234[From_3D_to_1D((cont + 1) * 2 - 1, cont2 * 2, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34
                            mapping_surf_3456[From_3D_to_1D((cont + 1) * 2 - 1, cont2, cont3 * 2, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34
                            contat_tot += 1
                            ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                            ind_c[contat_tot] = num_ele_12 + num_ele_34
                        end
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx
        for cont2 = 1:Ny
            cont3=1
            # for cont3 = 1
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont3 + 1 <= Nz
                            if grids[k2][cont][cont2][cont3+1]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && (!(externals_grids[k, 6][cont, cont2, cont3]) || check_om)
                        num_ele_56 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_56_se[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2, cont3 * 2 - 1, Nx, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_3456[From_3D_to_1D(cont * 2, cont2, cont3 * 2 - 1, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12 + num_ele_34 + num_ele_56
                        break
                    end
                end
            # end
        end
    end
    for cont = 1:Nx
        for cont2 = 1:Ny
            for cont3 = 2:Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont3 + 1 <= Nz
                            if grids[k2][cont][cont2][cont3+1]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !(grids[k][cont][cont2][cont3-1]) && (!externals_grids[k, 6][cont, cont2, cont3] || check_om)
                        num_ele_56 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_56_se[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2, cont3 * 2 - 1, Nx, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_3456[From_3D_to_1D(cont * 2, cont2, cont3 * 2 - 1, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12 + num_ele_34 + num_ele_56
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx
        for cont2 = 1:Ny
            for cont3 = Nz
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont3 - 1 >= 1
                            if grids[k2][cont][cont2][cont3-1]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && (!(externals_grids[k, 5][cont, cont2, cont3]) || check_om)
                        num_ele_56 += 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_56_se[From_3D_to_1D(cont, cont2, cont3 + 1, Nx, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2, (cont3 + 1) * 2 - 1, Nx, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34 + num_ele_56
                        mapping_surf_3456[From_3D_to_1D(cont * 2, cont2, (cont3 + 1) * 2 - 1, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                        contat_tot += 1
                        ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                        ind_c[contat_tot] = num_ele_12 + num_ele_34 + num_ele_56
                        break
                    end
                end
            end
        end
    end
    for cont = 1:Nx
        for cont2 = 1:Ny
            for cont3 = 1:Nz-1
                for k = 1:num_grids
                    check_om = false
                    for k2 = 1:num_grids
                        if k != k2 && cont3 - 1 >= 1
                            if grids[k2][cont][cont2][cont3-1]
                                check_om = true
                            end
                        end
                    end
                    if grids[k][cont][cont2][cont3] && !(grids[k][cont][cont2][cont3+1]) && (!externals_grids[k, 5][cont, cont2, cont3] || check_om)
                        controllo_altri = 0
                        for k2 = 1:num_grids
                            if k != k2
                                if grids[k2][cont][cont2][cont3+1]
                                    controllo_altri = 1
                                    break
                                end
                            end
                        end
                        if controllo_altri == 0
                            num_ele_56 += 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_56_se[From_3D_to_1D(cont, cont2, cont3 + 1, Nx, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                            mapping_surf_1256[From_3D_to_1D(cont, cont2 * 2, (cont3 + 1) * 2 - 1, Nx, 2 * (Ny + 1) - 1)] = num_ele_12 + num_ele_34 + num_ele_56
                            mapping_surf_3456[From_3D_to_1D(cont * 2, cont2, (cont3 + 1) * 2 - 1, 2 * (Nx + 1) - 1, Ny)] = num_ele_12 + num_ele_34 + num_ele_56
                            contat_tot += 1
                            ind_r[contat_tot] = bin_search(nodes[convert(Int64, map_volumes[p31])], nodes_red)
                            ind_c[contat_tot] = num_ele_12 + num_ele_34 + num_ele_56
                        end
                        break
                    end
                end
            end
        end
    end
    nnz_surf = num_ele_12 + num_ele_34 + num_ele_56
    ind_r = convert(Array{Int64}, ind_r)
    Gamma = sparse(ind_r[1:contat_tot], ind_c[1:contat_tot], ones(contat_tot), length(nodes_red), nnz_surf)
    exp_mat = Dict("N1" => 0, "N2" => 0, "N3" => 0, "exp_P" => Array{SparseMatrixCSC{Float64,Int64}}(undef, 3, 3))
    inn = findall(x -> x != 0, mapping_surf_12_se)
    N1 = length(inn)
    exp_mat["N1"] = N1
    exp_mat["exp_P"][1, 1] = sparse(inn, mapping_surf_12_se[inn], ones(N1), size(mapping_surf_12_se, 1), nnz_surf)
    inn = findall(x -> x != 0, mapping_surf_34_se)
    N2 = length(inn)
    exp_mat["N2"] = N2
    exp_mat["exp_P"][2, 2] = sparse(inn, mapping_surf_34_se[inn], ones(N2), size(mapping_surf_34_se, 1), nnz_surf)
    inn = findall(x -> x != 0, mapping_surf_56_se)
    N3 = length(inn)
    exp_mat["N3"] = N3
    exp_mat["exp_P"][3, 3] = sparse(inn, mapping_surf_56_se[inn], ones(N3), size(mapping_surf_56_se, 1), nnz_surf)
    inn1 = findall(x -> x <= num_ele_12, mapping_surf_1234)
    inn2 = findall(x -> x != 0, mapping_surf_1234)
    inn = intersect(inn1, inn2)
    N4 = length(inn)
    exp_mat["exp_P"][2, 1] = sparse(inn, mapping_surf_1234[inn], ones(N4), size(mapping_surf_1234, 1), nnz_surf)
    inn = findall(x -> x > num_ele_12, mapping_surf_1234)
    N5 = length(inn)
    exp_mat["exp_P"][1, 2] = sparse(inn, mapping_surf_1234[inn], ones(N5), size(mapping_surf_1234, 1), nnz_surf)
    inn1 = findall(x -> x <= num_ele_12, mapping_surf_1256)
    inn2 = findall(x -> x != 0, mapping_surf_1256)
    inn = intersect(inn1, inn2)
    N6 = length(inn)
    exp_mat["exp_P"][3, 1] = sparse(inn, mapping_surf_1256[inn], ones(N6), size(mapping_surf_1256, 1), nnz_surf)
    inn = findall(x -> x > num_ele_12 + num_ele_34, mapping_surf_1256)
    N7 = length(inn)
    exp_mat["exp_P"][1, 3] = sparse(inn, mapping_surf_1256[inn], ones(N7), size(mapping_surf_1256, 1), nnz_surf)
    inn1 = findall(x -> x <= num_ele_12 + num_ele_34, mapping_surf_3456)
    inn2 = findall(x -> x != 0, mapping_surf_3456)
    inn = intersect(inn1, inn2)
    N8 = length(inn)
    exp_mat["exp_P"][3, 2] = sparse(inn, mapping_surf_3456[inn], ones(N8), size(mapping_surf_3456, 1), nnz_surf)

    inn = findall(x -> x > num_ele_12 + num_ele_34, mapping_surf_3456)
    N9 = length(inn)
    exp_mat["exp_P"][2, 3] = sparse(inn, mapping_surf_3456[inn], ones(N9), size(mapping_surf_3456, 1), nnz_surf)

    return exp_mat, Gamma
end