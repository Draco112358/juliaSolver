include("From_3D_to_1D.jl")
include("bin_search.jl")
using LinearAlgebra

function create_nodes_ref(grids, num_full_vox, external_grids, mapping_vols, dominant_list)
    num_grids = size(external_grids, 1)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    nodes = zeros(num_full_vox)
    lista_nod = zeros(num_full_vox)
    cont_d = 0
    for ka in range(1,length(dominant_list))
        k = dominant_list[ka]
        for cont = 1:Nx
            for cont2 = 1:Ny
                for cont3 = 1:Nz
                    if grids[k][cont][cont2][cont3]
                        c1 = 1 + 2 * (cont - 1) + 1
                        c2 = 1 + 2 * (cont2 - 1) + 1
                        c3 = 1 + 2 * (cont3 - 1) + 1
                        f1 = external_grids[k,1][cont,cont2,cont3]
                        f2 = external_grids[k,2][cont,cont2,cont3]
                        f3 = external_grids[k,3][cont,cont2,cont3]
                        f4 = external_grids[k,4][cont,cont2,cont3]
                        f5 = external_grids[k,5][cont,cont2,cont3]
                        f6 = external_grids[k,6][cont,cont2,cont3]
                        f1_c = false
                        f2_c = false
                        f3_c = false
                        f4_c = false
                        f5_c = false
                        f6_c = false
                        if cont2 - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,2][cont,cont2 - 1,cont3]
                                        f1_c = true
                                    end
                                end
                            end
                        end
                        if cont2 + 1 <= Ny
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,1][cont,cont2 + 1,cont3]
                                        f2_c = true
                                    end
                                end
                            end
                        end
                        if cont - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,4][cont - 1,cont2,cont3]
                                        f3_c = true
                                    end
                                end
                            end
                        end
                        if cont + 1 <= Nx
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,3][cont + 1,cont2,cont3]
                                        f4_c = true
                                    end
                                end
                            end
                        end
                        if cont3 - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,6][cont,cont2,cont3 - 1]
                                        f5_c = true
                                    end
                                end
                            end
                        end
                        if cont3 + 1 <= Nz
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,5][cont,cont2,cont3 + 1]
                                        f6_c = true
                                    end
                                end
                            end
                        end
                        is_f1 = f1
                        is_f2 = f2
                        if f1 && f2
                            is_f1 = false
                            is_f2 = false
                            if f1_c && !f2_c
                                is_f1 = true
                            elseif !f1_c && f2_c
                                is_f2 = true
                            end
                        end
                        is_f3 = f3
                        is_f4 = f4
                        if f3 && f4
                            is_f3 = false
                            is_f4 = false
                            if f3_c && !f4_c
                                is_f3 = true
                            elseif !f3_c && f4_c
                                is_f4 = true
                            end
                        end
                        is_f5 = f5
                        is_f6 = f6
                        if f5 && f6
                            is_f5 = false
                            is_f6 = false
                            if f5_c && !f6_c
                                is_f5 = true
                            elseif !f5_c && f6_c
                                is_f6 = true
                            end
                        end
                        if any([is_f1 is_f2 is_f3 is_f4 is_f5 is_f6])
                            if is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 - 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && !is_f1 && !is_f3 && !is_f4 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 + 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f3 && !is_f1 && !is_f2 && !is_f4 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f4 && !is_f1 && !is_f2 && !is_f3 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f5 && !is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f6 && !is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f3 && !is_f2 && !is_f4 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 - 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f4 && !is_f2 && !is_f3 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 - 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f5 && !is_f2 && !is_f3 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 - 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f6 && !is_f2 && !is_f3 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 - 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f3 && is_f5 && !is_f2 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 - 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f3 && is_f6 && !is_f2 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 - 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f4 && is_f5 && !is_f2 && !is_f3 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 - 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f1 && is_f4 && is_f6 && !is_f2 && !is_f3 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 - 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f3 && !is_f1 && !is_f4 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 + 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f4 && !is_f1 && !is_f3 && !is_f5 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 + 1, c3, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f5 && !is_f1 && !is_f3 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 + 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f6 && !is_f1 && !is_f3 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2 + 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f3 && is_f5 && !is_f1 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 + 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f3 && is_f6 && !is_f1 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2 + 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f4 && is_f5 && !is_f1 && !is_f3 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 + 1, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f2 && is_f4 && is_f6 && !is_f1 && !is_f3 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2 + 1, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f3 && is_f5 && !is_f1 && !is_f2 && !is_f4 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f3 && is_f6 && !is_f1 && !is_f2 && !is_f4 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 - 1, c2, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f4 && is_f5 && !is_f1 && !is_f2 && !is_f3 && !is_f6
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2, c3 - 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            elseif is_f4 && is_f6 && !is_f1 && !is_f2 && !is_f3 && !is_f5
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1 + 1, c2, c3 + 1, 3 * Nx, 3 * Ny)
                                cont_d += 1
                                lista_nod[cont_d] = nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])]
                            end
                        else
                            nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3, 3 * Nx, 3 * Ny)
                        end
                    end
                end
            end
        end
    end
    nodes_red_dom = sort(unique(lista_nod[1:cont_d]))
    non_domin_list = setdiff(1:num_grids, dominant_list)
    nodes_reused = zeros(num_full_vox)
    cont_reu = 0
    for ka in range(1,length(non_domin_list))
        k = non_domin_list[ka]
        for cont = 1:Nx
            for cont2 = 1:Ny
                for cont3 = 1:Nz
                    if grids[k][cont][cont2][cont3]
                        c1 = 1 + 2 * (cont - 1) + 1
                        c2 = 1 + 2 * (cont2 - 1) + 1
                        c3 = 1 + 2 * (cont3 - 1) + 1
                        f1 = external_grids[k,1][cont,cont2,cont3]
                        f2 = external_grids[k,2][cont,cont2,cont3]
                        f3 = external_grids[k,3][cont,cont2,cont3]
                        f4 = external_grids[k,4][cont,cont2,cont3]
                        f5 = external_grids[k,5][cont,cont2,cont3]
                        f6 = external_grids[k,6][cont,cont2,cont3]
                        f1_c = false
                        f2_c = false
                        f3_c = false
                        f4_c = false
                        f5_c = false
                        f6_c = false
                        if cont2 - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,2][cont,cont2 - 1,cont3]
                                        f1_c = true
                                    end
                                end
                            end
                        end
                        if cont2 + 1 <= Ny
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,1][cont,cont2 + 1,cont3]
                                        f2_c = true
                                    end
                                end
                            end
                        end
                        if cont - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,4][cont - 1,cont2,cont3]
                                        f3_c = true
                                    end
                                end
                            end
                        end
                        if cont + 1 <= Nx
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,3][cont + 1,cont2,cont3]
                                        f4_c = true
                                    end
                                end
                            end
                        end
                        if cont3 - 1 > 1
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,6][cont,cont2,cont3 - 1]
                                        f5_c = true
                                    end
                                end
                            end
                        end
                        if cont3 + 1 <= Nz
                            for k2 = 1:num_grids
                                if k != k2
                                    if external_grids[k2,5][cont,cont2,cont3 + 1]
                                        f6_c = true
                                    end
                                end
                            end
                        end
                        is_f1 = f1
                        is_f2 = f2
                        if f1 && f2
                            is_f1 = false
                            is_f2 = false
                            if f1_c && !f2_c
                                is_f1 = true
                            elseif !f1_c && f2_c
                                is_f2 = true
                            end
                        end
                        is_f3 = f3
                        is_f4 = f4
                        if f3 && f4
                            is_f3 = false
                            is_f4 = false
                            if f3_c && !f4_c
                                is_f3 = true
                            elseif !f3_c && f4_c
                                is_f4 = true
                            end
                        end
                        is_f5 = f5
                        is_f6 = f6
                        if f5 && f6
                            is_f5 = false
                            is_f6 = false
                            if f5_c && !f6_c
                                is_f5 = true
                            elseif !f5_c && f6_c
                                is_f6 = true
                            end
                        end
                        if any([is_f1, is_f2, is_f3, is_f4, is_f5, is_f6])
                            nodes_to_see = build_nodes(c1, c2, c3, 3*Nx, 3*Ny)
                            nodo_shared, val_nodo = bin_search_mod(nodes_to_see, nodes_red_dom)
                            if abs(nodo_shared) > 1e-8
                                nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = val_nodo
                                cont_reu += 1
                                nodes_reused[cont_reu] = val_nodo
                            else
                                if is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2-1, c3, 3*Nx, 3*Ny)
                                elseif is_f2 && !is_f1 && !is_f3 && !is_f4 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2+1, c3, 3*Nx, 3*Ny)
                                elseif is_f3 && !is_f1 && !is_f2 && !is_f4 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2, c3, 3*Nx, 3*Ny)
                                elseif is_f4 && !is_f1 && !is_f2 && !is_f3 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2, c3, 3*Nx, 3*Ny)
                                elseif is_f5 && !is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3-1, 3*Nx, 3*Ny)
                                elseif is_f6 && !is_f1 && !is_f2 && !is_f3 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3+1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f3 && !is_f2 && !is_f4 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2-1, c3, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f4 && !is_f2 && !is_f3 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2-1, c3, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f5 && !is_f2 && !is_f3 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2-1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f6 && !is_f2 && !is_f3 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2-1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f3 && is_f5 && !is_f2 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2-1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f3 && is_f6 && !is_f2 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2-1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f4 && is_f5 && !is_f2 && !is_f3 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2-1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f1 && is_f4 && is_f6 && !is_f2 && !is_f3 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2-1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f3 && !is_f1 && !is_f4 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2+1, c3, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f4 && !is_f1 && !is_f3 && !is_f5 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2+1, c3, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f5 && !is_f1 && !is_f3 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2+1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f6 && !is_f1 && !is_f3 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2+1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f3 && is_f5 && !is_f1 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2+1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f3 && is_f6 && !is_f1 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2+1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f4 && is_f5 && !is_f1 && !is_f3 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2+1, c3-1, 3*Nx, 3*Ny)
                                elseif is_f2 && is_f4 && is_f6 && !is_f1 && !is_f3 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2+1, c3+1, 3*Nx, 3*Ny)
                                elseif is_f3 && is_f5 && !is_f1 && !is_f2 && !is_f4 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2, c3-1, 3*Nx, 3*Ny)
                                elseif is_f3 && is_f6 && !is_f1 && !is_f2 && !is_f4 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1-1, c2, c3+1, 3*Nx, 3*Ny)
                                elseif is_f4 && is_f5 && !is_f1 && !is_f2 && !is_f3 && !is_f6
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2, c3-1, 3*Nx, 3*Ny)
                                elseif is_f4 && is_f6 && !is_f1 && !is_f2 && !is_f3 && !is_f5
                                    nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1+1, c2, c3+1, 3*Nx, 3*Ny)
                                end
                            end
                        else
                            nodes[convert(Int64,mapping_vols[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)])] = From_3D_to_1D(c1, c2, c3, 3*Nx, 3*Ny)
                        end
                    end
                end
            end
        end
    end

    nodes_red = sort(unique(nodes))
    nodes_reused_clean = sort(unique(nodes_reused[1:cont_reu]))

    return nodes,nodes_red,nodes_reused_clean
end
    
function  bin_search_mod(nodes_to_see, nodes_red_dom)
    val_nodo = 0
    nodo_shared = 0
    for k = 1:26
        nodo_shared = bin_search(nodes_to_see[k], nodes_red_dom)
        if abs(nodo_shared) > 1e-8
            val_nodo = nodes_to_see[k]
            break
        end
    end
    return nodo_shared, val_nodo
end

function build_nodes(c1, c2, c3, Nx, Ny)
    nodes = zeros(26)
    k = 1
    nodes[k] = From_3D_to_1D(c1-1, c2-1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2-1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2-1, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2-1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2-1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2-1, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2-1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2-1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2-1, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2+1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2+1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1-1, c2+1, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2+1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2+1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1, c2+1, c3+1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2+1, c3-1, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2+1, c3, Nx, Ny)
    k += 1
    nodes[k] = From_3D_to_1D(c1+1, c2+1, c3+1, Nx, Ny)

    return nodes
end            