using LinearAlgebra
include("From_3D_to_1D.jl")

function build_center_P_Voxels(min_v, Nx, Ny, Nz, sx, sy, sz)
    centri_vox = Dict()
    centri_vox["p12_se"] = zeros((Nx) * (Ny + 1) * (Nz), 3)
    centri_vox["p34_se"] = zeros((Nx + 1) * (Ny) * (Nz), 3)
    centri_vox["p56_se"] = zeros((Nx) * (Ny) * (Nz + 1), 3)
    centri_vox["p1234"] = zeros((2 * (Nx + 1) - 1) * (2 * (Ny + 1) - 1) * (Nz), 3)
    centri_vox["p1256"] = zeros((Nx) * (2 * (Ny + 1) - 1) * (2 * (Nz + 1) - 1), 3)
    centri_vox["p3456"] = zeros((2 * (Nx + 1) - 1) * (Ny) * (2 * (Nz + 1) - 1), 3)
    for cont in CartesianIndices((1:Nx+1, 1:Ny+1, 1:Nz+1))
        cx = min_v[1] + sx * (cont[1] - 1) + sx / 2
        cy = min_v[2] + sy * (cont[2] - 1) + sy / 2
        cz = min_v[3] + sz * (cont[3] - 1) + sz / 2
        if cont[2] != Ny + 1 && cont[1] != Nx + 1
            centri_vox["p56_se"][from_3D_to_1D(cont[1], cont[2], cont[3], Nx, Ny), :] = [cx cy cz]
        end
        if cont[3] != Nz + 1 && cont[1] != Nx + 1
            centri_vox["p12_se"][from_3D_to_1D(cont[1], cont[2], cont[3], Nx, Ny + 1), :] = [cx cy cz]
        end
        if cont[3] != Nz + 1 && cont[2] != Ny + 1
            centri_vox["p34_se"][from_3D_to_1D(cont[1], cont[2], cont[3], Nx + 1, Ny), :] = [cx cy cz]
        end
    end
    # Threads.@threads for cont=1:Nx
    #     for cont2=1:Ny+1
    #         for cont3=1:Nz
    #             cx=min_v[1]+sx*(cont-1)+sx/2
    #             cy=min_v[2]+sy*(cont2-1)+sy/2
    #             cz=min_v[3]+sz*(cont3-1)+sz/2
    #             @inbounds centri_vox["p12_se"][from_3D_to_1D(cont, cont2, cont3, Nx, Ny+1),:] = [cx cy cz]
    #         end
    #     end
    # end
    # Threads.@threads for cont=1:Nx+1
    #     for cont2=1:Ny
    #         for cont3=1:Nz
    #             sum +=1
    #             cx=min_v[1]+sx*(cont-1)+sx/2
    #             cy=min_v[2]+sy*(cont2-1)+sy/2
    #             cz=min_v[3]+sz*(cont3-1)+sz/2
    #             @inbounds centri_vox["p34_se"][from_3D_to_1D(cont, cont2, cont3, Nx+1, Ny),:] = [cx cy cz]
    #         end
    #     end
    # end
    # Threads.@threads for cont=1:Nx
    #     for cont2=1:Ny
    #         for cont3=1:Nz+1
    #             sum +=1
    #             cx=min_v[1]+sx*(cont-1)+sx/2
    #             cy=min_v[2]+sy*(cont2-1)+sy/2
    #             cz=min_v[3]+sz*(cont3-1)+sz/2
    #             @inbounds centri_vox["p56_se"][from_3D_to_1D(cont, cont2, cont3, Nx, Ny),:] = [cx cy cz]
    #         end
    #     end
    # end
    for cont in CartesianIndices((1:2*(Nx+1)-1, 1:2*(Ny+1)-1, 1:2*(Nz+1)-1))
        if cont[3] <= Nz
            cx = min_v[1] + sx * (ceil(cont[1] / 2) - 1) + sx / 2
            cy = min_v[2] + sy * (ceil(cont[2] / 2) - 1) + sy / 2
            cz = min_v[3] + sz * (cont[3] - 1) + sz / 2
            centri_vox["p1234"][from_3D_to_1D(cont[1], cont[2], cont[3], 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1), :] = [cx cy cz]
        end
        if cont[2] <= Ny
            cx = min_v[1] + sx * (ceil(cont[1] / 2) - 1) + sx / 2
            cy = min_v[2] + sy * (cont[2] - 1) + sy / 2
            cz = min_v[3] + sz * (ceil(cont[3] / 2) - 1) + sz / 2
            centri_vox["p3456"][from_3D_to_1D(cont[1], cont[2], cont[3], 2 * (Nx + 1) - 1, Ny), :] = [cx cy cz]
        end
        if cont[1] <= Nx
            cx = min_v[1] + sx * (cont[1] - 1) + sx / 2
            cy = min_v[2] + sy * (ceil(cont[2] / 2) - 1) + sy / 2
            cz = min_v[3] + sz * (ceil(cont[3] / 2) - 1) + sz / 2
            centri_vox["p1256"][from_3D_to_1D(cont[1], cont[2], cont[3], Nx, 2 * (Ny + 1) - 1), :] = [cx cy cz]
        end
    end
    # for cont = 1:2*(Nx+1)-1
    #     for cont2 = 1:2*(Ny+1)-1
    #         for cont3 = 1:Nz
    #             cx = min_v[1] + sx * (ceil(cont / 2) - 1) + sx / 2
    #             cy = min_v[2] + sy * (ceil(cont2 / 2) - 1) + sy / 2
    #             cz = min_v[3] + sz * (cont3 - 1) + sz / 2
    #             @inbounds centri_vox["p1234"][from_3D_to_1D(cont, cont2, cont3, 2 * (Nx + 1) - 1, 2 * (Ny + 1) - 1), :] = [cx cy cz]
    #         end
    #     end
    # end
    # for cont = 1:Nx
    #     for cont2 = 1:2*(Ny+1)-1
    #         for cont3 = 1:2*(Nz+1)-1
    #             cx = min_v[1] + sx * (cont - 1) + sx / 2
    #             cy = min_v[2] + sy * (ceil(cont2 / 2) - 1) + sy / 2
    #             cz = min_v[3] + sz * (ceil(cont3 / 2) - 1) + sz / 2
    #             @inbounds centri_vox["p1256"][from_3D_to_1D(cont, cont2, cont3, Nx, 2 * (Ny + 1) - 1), :] = [cx cy cz]
    #         end
    #     end
    # end
    # for cont = 1:2*(Nx+1)-1
    #     for cont2 = 1:Ny
    #         for cont3 = 1:2*(Nz+1)-1
    #             cx = min_v[1] + sx * (ceil(cont / 2) - 1) + sx / 2
    #             cy = min_v[2] + sy * (cont2 - 1) + sy / 2
    #             cz = min_v[3] + sz * (ceil(cont3 / 2) - 1) + sz / 2
    #             @inbounds centri_vox["p3456"][from_3D_to_1D(cont, cont2, cont3, 2 * (Nx + 1) - 1, Ny), :] = [cx cy cz]
    #         end
    #     end
    # end
    return centri_vox
end