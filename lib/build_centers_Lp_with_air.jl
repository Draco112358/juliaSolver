include("From_3D_to_1D.jl")

function build_centers_Lp_with_air(grids, centri_Vox_with_air)
    Nx = size(grids[1],1)
    Ny = size(grids[1][1],1)
    Nz = size(grids[1][1][1],1)
    Cx = zeros((Nx-1)*Ny*Nz, 3)
    for cont3 = 1:Nz
        for cont2 = 1:Ny
            for cont = 1:Nx-1
                pos = From_3D_to_1D(cont, cont2, cont3, Nx-1, Ny)
                A = centri_Vox_with_air[From_3D_to_1D(cont, cont2, cont3, Nx, Ny), :]
                B = centri_Vox_with_air[From_3D_to_1D(cont+1, cont2, cont3, Nx, Ny), :]
                Cx[pos, :] = 0.5*(A+B)
            end
        end
    end
    Cy = zeros((Ny-1)*Nx*Nz, 3)
    for cont3 = 1:Nz
        for cont = 1:Nx
            for cont2 = 1:Ny-1
                pos = From_3D_to_1D(cont, cont2, cont3, Nx, Ny-1)
                A = centri_Vox_with_air[From_3D_to_1D(cont, cont2, cont3, Nx, Ny), :]
                B = centri_Vox_with_air[From_3D_to_1D(cont, cont2+1, cont3, Nx, Ny), :]
                Cy[pos, :] = 0.5*(A+B)
            end
        end
    end
    Cz = zeros((Nz-1)*Nx*Ny, 3)
    for cont = 1:Nx
        for cont2 = 1:Ny
            for cont3 = 1:Nz-1
                pos = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                A = centri_Vox_with_air[From_3D_to_1D(cont, cont2, cont3, Nx, Ny), :]
                B = centri_Vox_with_air[From_3D_to_1D(cont, cont2, cont3+1, Nx, Ny), :]
                Cz[pos, :] = 0.5*(A+B)
            end
        end
    end
    return Cx, Cy, Cz
end
