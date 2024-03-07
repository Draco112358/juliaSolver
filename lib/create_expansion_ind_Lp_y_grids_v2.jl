include("From_3D_to_1D.jl")
include("bin_search.jl")

function create_expansion_ind_Lp_y_grids_v2(grids, map, l_ind, mapping_Vox, nodes, nodes_red)
    num_grids = length(grids)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    Ind_out = zeros(Int64,l_ind, 2)
    pos = 0
    for cont3 in 1:Nz
        for cont in 1:Nx
            for cont2 in 1:Ny-1
                for k in 1:num_grids
                    if grids[k][cont][cont2][cont3] && grids[k][cont][cont2+1][cont3]
                        nn1=bin_search(nodes[convert(Int64,mapping_Vox[from_3D_to_1D(cont, cont2, cont3, Nx, Ny)])],nodes_red);
                        nn2=bin_search(nodes[convert(Int64,mapping_Vox[from_3D_to_1D(cont, cont2+1, cont3, Nx, Ny)])],nodes_red);
                        if abs(nn1-nn2) > 1e-8
                            pos += 1
                            Ind_out[pos, 1] = from_3D_to_1D(cont, cont2, cont3, Nx, Ny-1)
                            Ind_out[pos, 2] = map[from_3D_to_1D(cont, cont2, cont3, Nx, Ny-1)]
                            break
                        end
                    end
                end
            end
        end
    end
    return Ind_out
end
