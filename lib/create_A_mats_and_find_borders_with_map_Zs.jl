include("From_3D_to_1D.jl")
include("bin_search.jl")
using SparseArrays

function create_A_mats_and_find_borders_with_map_Zs(grids, mapping_Vox, mapAx, NAx, mapAy, NAy, mapAz, NAz, materials, nodes, nodes_red, nodes_reused_clean)
    num_grids = length(grids)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    lix_mat = zeros(NAx, 2)
    lix_border = zeros(NAx, 2)
    rAx = zeros(2 * NAx)
    cAx = zeros(2 * NAx)
    vAx = zeros(2 * NAx)
    maps_Zs = Dict("x_xy" => zeros(NAx, 4), "x_zx" => zeros(NAx, 4), "y_xy" => zeros(NAy, 4), "y_yz" => zeros(NAy, 4), "z_zx" => zeros(NAz, 4), "z_yz" => zeros(NAz, 4))
    num_ele = 0
    for cont3 = 1:Nz
        for cont2 = 1:Ny
            for cont = 1:Nx - 1
                for k = 1:num_grids
                    if grids[k][cont][cont2][cont3] && grids[k][cont + 1][cont2][cont3]
                        nodo_v1 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                        nodo_v2 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont + 1, cont2, cont3, Nx, Ny)])]
                        if abs(nodo_v1 - nodo_v2) > 1e-8
                            nn1 = bin_search(nodo_v1, nodes_red)
                            nn2 = bin_search(nodo_v2, nodes_red)
                            pos = convert(Int64,mapAx[From_3D_to_1D(cont, cont2, cont3, Nx - 1, Ny)])
                            num_ele += 1
                            rAx[num_ele] = pos
                            cAx[num_ele] = nn1
                            vAx[num_ele] = -1
                            num_ele += 1
                            rAx[num_ele] = pos
                            cAx[num_ele] = nn2
                            vAx[num_ele] = 1
                            lix_mat[pos, :] = [k, k]
                            if materials[k]["sigmar"] != 0
                                if cont3 == 1 || cont3 == Nz
                                    maps_Zs["x_xy"][pos, 1] += 1
                                    maps_Zs["x_xy"][pos, 2] += 1
                                else
                                    if cont3 > 1
                                        if !grids[k][cont][cont2][cont3 - 1]
                                            maps_Zs["x_xy"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2][cont3 - 1]
                                                    maps_Zs["x_xy"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2][cont3 - 1]
                                            maps_Zs["x_xy"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2][cont3 - 1]
                                                    maps_Zs["x_xy"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont3 < Nz
                                        if !grids[k][cont][cont2][cont3 + 1]
                                            maps_Zs["x_xy"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2][cont3 + 1]
                                                    maps_Zs["x_xy"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2][cont3 + 1]
                                            maps_Zs["x_xy"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2][cont3 + 1]
                                                    maps_Zs["x_xy"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                                if cont2 == 1 || cont2 == Ny
                                    maps_Zs["x_zx"][pos, 1] += 1
                                    maps_Zs["x_zx"][pos, 2] += 1
                                else
                                    if cont2 > 1
                                        if !grids[k][cont][cont2 - 1][cont3]
                                            maps_Zs["x_zx"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 - 1][cont3]
                                                    maps_Zs["x_zx"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2 - 1][cont3]
                                            maps_Zs["x_zx"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2 - 1][cont3]
                                                    maps_Zs["x_zx"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont2 < Ny
                                        if !grids[k][cont][cont2 + 1][cont3]
                                            maps_Zs["x_zx"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 + 1][cont3]
                                                    maps_Zs["x_zx"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2 + 1][cont3]
                                            maps_Zs["x_zx"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2 + 1][cont3]
                                                    maps_Zs["x_zx"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            if cont > 1
                                if !grids[k][cont - 1][cont2][cont3]
                                    lix_border[pos, 1] = k
                                end
                            else
                                lix_border[pos, 1] = k
                            end
                            if cont + 1 == Nx
                                lix_border[pos, 2] = k
                            elseif cont + 2 <= Nx
                                if !grids[k][cont + 2][cont2][cont3]
                                    lix_border[pos, 2] = k
                                end
                            end
                            break
                        end
                    end
                end
            end
        end
    end

    liy_mat = zeros(NAy, 2)
    liy_border = zeros(NAy, 2)
    rAy = zeros(2 * NAy)
    cAy = zeros(2 * NAy)
    vAy = zeros(2 * NAy)
    num_ele = 0
    for cont3 = 1:Nz
        for cont = 1:Nx
            for cont2 = 1:Ny - 1
                for k = 1:num_grids
                    if grids[k][cont][cont2][cont3] && grids[k][cont][cont2 + 1][cont3]
                        nodo_v1 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                        nodo_v2 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont, cont2 + 1, cont3, Nx, Ny)])]
                        if abs(nodo_v1 - nodo_v2) > 1e-8
                            nn1 = bin_search(nodo_v1, nodes_red)
                            nn2 = bin_search(nodo_v2, nodes_red)
                            pos = convert(Int64,mapAy[From_3D_to_1D(cont, cont2, cont3, Nx, Ny - 1)])
                            num_ele += 1
                            rAy[num_ele] = pos
                            cAy[num_ele] = nn1
                            vAy[num_ele] = -1
                            num_ele += 1
                            rAy[num_ele] = pos
                            cAy[num_ele] = nn2
                            vAy[num_ele] = 1
                            liy_mat[pos, :] = [k, k]
                            if materials[k]["sigmar"] != 0
                                if cont3 == 1 || cont3 == Nz
                                    maps_Zs["y_xy"][pos, 1] += 1
                                    maps_Zs["y_xy"][pos, 2] += 1
                                else
                                    if cont3 > 1
                                        if !grids[k][cont][cont2][cont3 - 1]
                                            maps_Zs["y_xy"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2][cont3 - 1]
                                                    maps_Zs["y_xy"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont][cont2 + 1][cont3 - 1]
                                            maps_Zs["y_xy"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 + 1][cont3 - 1]
                                                    maps_Zs["y_xy"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont3 < Nz
                                        if !grids[k][cont][cont2][cont3 + 1]
                                            maps_Zs["y_xy"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2][cont3 + 1]
                                                    maps_Zs["y_xy"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont][cont2 + 1][cont3 + 1]
                                            maps_Zs["y_xy"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 + 1][cont3 + 1]
                                                    maps_Zs["y_xy"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                                if cont == 1 || cont == Nx
                                    maps_Zs["y_yz"][pos, 1] += 1
                                    maps_Zs["y_yz"][pos, 2] += 1
                                else
                                    if cont > 1
                                        if !grids[k][cont - 1][cont2][cont3]
                                            maps_Zs["y_yz"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont - 1][cont2][cont3]
                                                    maps_Zs["y_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont - 1][cont2 + 1][cont3]
                                            maps_Zs["y_yz"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont - 1][cont2 + 1][cont3]
                                                    maps_Zs["y_yz"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont < Nx
                                        if !grids[k][cont + 1][cont2][cont3]
                                            maps_Zs["y_yz"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2][cont3]
                                                    maps_Zs["y_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2 + 1][cont3]
                                            maps_Zs["y_yz"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2 + 1][cont3]
                                                    maps_Zs["y_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            if cont2 > 1
                                if !grids[k][cont][cont2 - 1][cont3]
                                    liy_border[pos, 1] = k
                                end
                            else
                                liy_border[pos, 1] = k
                            end
                            if cont2 + 1 == Ny
                                liy_border[pos, 2] = k
                            elseif cont2 + 2 <= Ny
                                if !grids[k][cont][cont2 + 2][cont3]
                                    liy_border[pos, 2] = k
                                end
                            end
                            break
                        end
                    end
                end
            end
        end
    end
    liz_mat = zeros(NAz, 2)
    liz_border = zeros(NAz, 2)
    rAz = zeros(2 * NAz)
    cAz = zeros(2 * NAz)
    vAz = zeros(2 * NAz)
    num_ele = 0
    for cont = 1:Nx
        for cont2 = 1:Ny
            for cont3 = 1:Nz - 1
                for k = 1:num_grids
                    if grids[k][cont][cont2][cont3] && grids[k][cont][cont2][cont3 + 1]
                        nodo_v1 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                        nodo_v2 = nodes[convert(Int64,mapping_Vox[From_3D_to_1D(cont, cont2, cont3 + 1, Nx, Ny)])]
                        if abs(nodo_v1 - nodo_v2) > 1e-8
                            nn1 = bin_search(nodo_v1, nodes_red)
                            nn2 = bin_search(nodo_v2, nodes_red)
                            pos = convert(Int64,mapAz[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])
                            num_ele += 1
                            rAz[num_ele] = pos
                            cAz[num_ele] = nn1
                            vAz[num_ele] = -1
                            num_ele += 1
                            rAz[num_ele] = pos
                            cAz[num_ele] = nn2
                            vAz[num_ele] = 1
                            liz_mat[pos, :] = [k, k]
                            if materials[k]["sigmar"] != 0
                                if cont == 1 || cont == Nx
                                    maps_Zs["z_zx"][pos, 1] += 1
                                    maps_Zs["z_zx"][pos, 2] += 1
                                else
                                    if cont > 1
                                        if !grids[k][cont - 1][cont2][cont3]
                                            maps_Zs["z_zx"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont - 1][cont2][cont3]
                                                    maps_Zs["z_zx"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont - 1][cont2][cont3 + 1]
                                            maps_Zs["z_zx"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont - 1][cont2][cont3 + 1]
                                                    maps_Zs["z_zx"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont < Nx
                                        if !grids[k][cont + 1][cont2][cont3]
                                            maps_Zs["z_zx"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2][cont3]
                                                    maps_Zs["z_zx"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont + 1][cont2][cont3]
                                            maps_Zs["z_zx"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont + 1][cont2][cont3]
                                                    maps_Zs["z_zx"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                                if cont2 == 1 || cont2 == Ny
                                    maps_Zs["z_yz"][pos, 1] += 1
                                    maps_Zs["z_yz"][pos, 2] += 1
                                else
                                    if cont2 > 1
                                        if !grids[k][cont][cont2 - 1][cont3]
                                            maps_Zs["z_yz"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 - 1][cont3]
                                                    maps_Zs["z_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont][cont2 - 1][cont3 + 1]
                                            maps_Zs["z_yz"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 - 1][cont3 + 1]
                                                    maps_Zs["z_yz"][pos, 4] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                    if cont2 < Ny
                                        if !grids[k][cont][cont2 + 1][cont3]
                                            maps_Zs["z_yz"][pos, 1] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 + 1][cont3]
                                                    maps_Zs["z_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                        if !grids[k][cont][cont2 + 1][cont3]
                                            maps_Zs["z_yz"][pos, 2] += 1
                                            for k2 = 1:num_grids
                                                if k != k2 && grids[k2][cont][cont2 + 1][cont3]
                                                    maps_Zs["z_yz"][pos, 3] = k2
                                                    break
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            if cont3 > 1
                                if !grids[k][cont][cont2][cont3 - 1]
                                    liz_border[pos, 1] = k
                                end
                            else
                                liz_border[pos, 1] = k
                            end
                            if cont3 + 1 == Nz
                                liz_border[pos, 2] = k
                            elseif cont3 + 2 <= Nz
                                if !grids[k][cont][cont2][cont3 + 2]
                                    liz_border[pos, 2] = k
                                end
                            end
                            break
                        end
                    end
                end
            end
        end
    end
    
    A = sparse([rAx; rAy.+NAx; rAz.+NAx.+NAy], [cAx; cAy; cAz], [vAx; vAy; vAz], NAx+NAy+NAz, length(nodes_red))
    return A,lix_mat,liy_mat,liz_mat,lix_border,liy_border,liz_border,maps_Zs
end

# using SparseArrays

# A = sparse([rAx; rAy+NAx; rAz+NAx+NAy], [cAx; cAy; cAz], [vAx; vAy; vAz], NAx+NAy+NAz, length(nodes_red))
