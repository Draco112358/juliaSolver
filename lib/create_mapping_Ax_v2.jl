include("From_3D_to_1D.jl")
include("bin_search.jl")
using Base.Iterators

function create_mapping_Ax_v2(grids, mapping_Vox, nodes, nodes_red)
    num_grids = length(grids)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    N_max = (Nx-1)*Ny*Nz
    mapping = zeros(Int64, N_max)
    num_ele = 0
    for cont2=1:Ny
        for cont3=1:Nz
            for cont=1:Nx-1
                for k in 1:num_grids
                    if grids[k][cont][cont2][cont3] && grids[k][cont+1][cont2][cont3]
                        nn1 = bin_search(nodes[convert(Int64,mapping_Vox[from_3D_to_1D(cont, cont2, cont3, Nx, Ny)])], nodes_red)
                        nn2 = bin_search(nodes[convert(Int64,mapping_Vox[from_3D_to_1D(cont+1, cont2, cont3, Nx, Ny)])], nodes_red)
                        if abs(nn1-nn2) > 1e-8
                            kkey = from_3D_to_1D(cont, cont2, cont3, Nx-1, Ny)
                            if mapping[kkey] == 0
                                num_ele += 1
                                mapping[kkey] = num_ele
                            end
                            break
                        end
                    end
                end
            end
        end
    end
    return mapping, num_ele
end