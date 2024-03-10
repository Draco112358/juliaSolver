using LinearAlgebra
include("From_3D_to_1D.jl")

function create_volume_centers(grids, map, num_centri, sx, sy, sz, min_v)
    centri_vox = zeros(num_centri, 3)
    id_mat = zeros(num_centri, 1)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    num_grids = length(grids)
    Threads.@threads for cont3=1:Nz
        for cont2=1:Ny
            for cont=1:Nx
                for k=1:num_grids
                    if grids[k][cont][cont2][cont3] != 0
                        pos = convert(Int64,map[from_3D_to_1D(cont, cont2, cont3, Nx, Ny)])
                        cx = min_v[1] + sx * (cont - 1) + sx / 2
                        cy = min_v[2] + sy * (cont2 - 1) + sy / 2
                        cz = min_v[3] + sz * (cont3 - 1) + sz / 2
                        @inbounds centri_vox[pos, :] = [cx cy cz]
                        @inbounds id_mat[pos] = k
                        break
                    end
                end
            end
        end
    end
    return centri_vox, id_mat
end