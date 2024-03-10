function create_Grids_externals(grids)
    Nx = size(grids[1], 1)
    Ny = size(grids[1][1], 1)
    Nz = size(grids[1][1][1], 1)
    num_grids = length(grids)
    OUTPUTgrids = Array{Array{Bool}}(undef, num_grids, 6)
    for k = 1:num_grids
        OUTPUTgrids[k, 1] = falses(Nx, Ny, Nz)
        OUTPUTgrids[k, 2] = falses(Nx, Ny, Nz)
        OUTPUTgrids[k, 3] = falses(Nx, Ny, Nz)
        OUTPUTgrids[k, 4] = falses(Nx, Ny, Nz)
        OUTPUTgrids[k, 5] = falses(Nx, Ny, Nz)
        OUTPUTgrids[k, 6] = falses(Nx, Ny, Nz)
    end
    for cont3 = 1:Nz
        cont = Nx # da eliminare quando si sarà risolta la questione del for senza range
        for cont2 = 1:Ny
            for k = 1:num_grids
                if (grids[k][1][cont2][cont3])
                    OUTPUTgrids[k, 3][1, cont2, cont3] = true
                    if Nx > 1
                        if !(grids[k][2][cont2][cont3])
                            OUTPUTgrids[k, 4][1, cont2, cont3] = true
                        end
                    end
                end
                if (grids[k][cont][cont2][cont3])
                    OUTPUTgrids[k, 4][cont, cont2, cont3] = true
                    if Nx > 1
                        if !(grids[k][cont-1][cont2][cont3])
                            OUTPUTgrids[k, 3][cont, cont2, cont3] = true
                        end
                    end
                end
            end
        end
    end
    # Threads.@threads for cont3 = 1:Nz
    #     for cont2 = 1:Ny
    #         for k = 1:num_grids
    #             if (grids[k][1][cont2][cont3])
    #                 OUTPUTgrids[k, 3][1, cont2, cont3] = true
    #                 if Nx > 1
    #                     if !(grids[k][2][cont2][cont3])
    #                         OUTPUTgrids[k, 4][1, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    # # for cont=Nx
    # Threads.@threads for cont3 = 1:Nz
    #     cont = Nx # da eliminare quando si sarà risolta la questione del for senza range
    #     for cont2 = 1:Ny
    #         for k = 1:num_grids
    #             if (grids[k][cont][cont2][cont3])
    #                 OUTPUTgrids[k, 4][cont, cont2, cont3] = true
    #                 if Nx > 1
    #                     if !(grids[k][cont-1][cont2][cont3])
    #                         OUTPUTgrids[k, 3][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # # end
    for cont3 = 1:Nz
        cont2 = Ny # da eliminare quando si sarà risolta la questione del for senza range
        for cont = 1:Nx
            for k = 1:num_grids
                if (grids[k][cont][1][cont3])
                    OUTPUTgrids[k, 1][cont, 1, cont3] = true
                    if Ny > 1
                        if !(grids[k][cont][2][cont3])
                            OUTPUTgrids[k, 2][cont, 1, cont3] = true
                        end
                    end
                end
                if (grids[k][cont][cont2][cont3])
                    OUTPUTgrids[k, 2][cont, cont2, cont3] = true
                    if Ny > 1
                        if !(grids[k][cont][cont2-1][cont3])
                            OUTPUTgrids[k, 1][cont, cont2, cont3] = true
                        end
                    end
                end
            end
        end
    end
    # Threads.@threads for cont3 = 1:Nz
    #     for cont = 1:Nx
    #         for k = 1:num_grids
    #             if (grids[k][cont][1][cont3])
    #                 OUTPUTgrids[k, 1][cont, 1, cont3] = true
    #                 if Ny > 1
    #                     if !(grids[k][cont][2][cont3])
    #                         OUTPUTgrids[k, 2][cont, 1, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # Threads.@threads for cont3 = 1:Nz
    #     cont2 = Ny # da eliminare quando si sarà risolta la questione del for senza range
    #     # for cont2=Ny
    #     for cont = 1:Nx
    #         for k = 1:num_grids
    #             if (grids[k][cont][cont2][cont3])
    #                 OUTPUTgrids[k, 2][cont, cont2, cont3] = true
    #                 if Ny > 1
    #                     if !(grids[k][cont][cont2-1][cont3])
    #                         OUTPUTgrids[k, 1][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    #     # end
    # end
    for cont2 = 1:Ny
        cont3 = Nz # da eliminare quando si sarà risolta la questione del for senza range
        for cont = 1:Nx
            for k = 1:num_grids
                if (grids[k][cont][cont2][1])
                    OUTPUTgrids[k, 5][cont, cont2, 1] = true
                    if Nz > 1
                        if !(grids[k][cont][cont2][2])
                            OUTPUTgrids[k, 6][cont, cont2, 1] = true
                        end
                    end
                end
                if (grids[k][cont][cont2][cont3])
                    OUTPUTgrids[k, 6][cont, cont2, cont3] = true
                    if Nz > 1
                        if !(grids[k][cont][cont2][cont3-1])
                            OUTPUTgrids[k, 5][cont, cont2, cont3] = true
                        end
                    end
                end
            end
        end
    end
    # Threads.@threads for cont2 = 1:Ny
    #     for cont = 1:Nx
    #         for k = 1:num_grids
    #             if (grids[k][cont][cont2][1])
    #                 OUTPUTgrids[k, 5][cont, cont2, 1] = true
    #                 if Nz > 1
    #                     if !(grids[k][cont][cont2][2])
    #                         OUTPUTgrids[k, 6][cont, cont2, 1] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # # for cont3 = Nz
    # for cont2 = 1:Ny
    #     cont3 = Nz # da eliminare quando si sarà risolta la questione del for senza range
    #     for cont = 1:Nx
    #         for k = 1:num_grids
    #             if (grids[k][cont][cont2][cont3])
    #                 OUTPUTgrids[k, 6][cont, cont2, cont3] = true
    #                 if Nz > 1
    #                     if !(grids[k][cont][cont2][cont3-1])
    #                         OUTPUTgrids[k, 5][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # # end
    for cont3 = 1:Nz
        for cont2 = 1:Ny
            for cont = 1:Nx
                for k = 1:num_grids
                    if (grids[k][cont][cont2][cont3])
                        if cont != 1 && cont != Nx
                            if !(grids[k][cont-1][cont2][cont3])
                                OUTPUTgrids[k, 3][cont, cont2, cont3] = true
                            end
                            if !(grids[k][cont+1][cont2][cont3])
                                OUTPUTgrids[k, 4][cont, cont2, cont3] = true
                            end
                        end
                        if cont2 != 1 && cont2 != Ny
                            if !(grids[k][cont][cont2-1][cont3])
                                OUTPUTgrids[k, 1][cont, cont2, cont3] = true
                            end
                            if !(grids[k][cont][cont2+1][cont3])
                                OUTPUTgrids[k, 2][cont, cont2, cont3] = true
                            end
                        end
                        if cont3 != 1 && cont3 != Nz
                            if !(grids[k][cont][cont2][cont3-1])
                                OUTPUTgrids[k, 5][cont, cont2, cont3] = true
                            end
                            if !(grids[k][cont][cont2][cont3+1])
                                OUTPUTgrids[k, 6][cont, cont2, cont3] = true
                            end
                        end
                    end
                end
            end
        end
    end
    # Threads.@threads for cont3 = 1:Nz
    #     for cont2 = 1:Ny
    #         for cont = 2:Nx-1
    #             for k = 1:num_grids
    #                 if (grids[k][cont][cont2][cont3])
    #                     if !(grids[k][cont-1][cont2][cont3])
    #                         OUTPUTgrids[k, 3][cont, cont2, cont3] = true
    #                     end
    #                     if !(grids[k][cont+1][cont2][cont3])
    #                         OUTPUTgrids[k, 4][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # Threads.@threads for cont3 = 1:Nz
    #     for cont2 = 2:Ny-1
    #         for cont = 1:Nx
    #             for k = 1:num_grids
    #                 if (grids[k][cont][cont2][cont3])
    #                     if !(grids[k][cont][cont2-1][cont3])
    #                         OUTPUTgrids[k, 1][cont, cont2, cont3] = true
    #                     end
    #                     if !(grids[k][cont][cont2+1][cont3])
    #                         OUTPUTgrids[k, 2][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # Threads.@threads for cont3 = 2:Nz-1
    #     for cont2 = 1:Ny
    #         for cont = 1:Nx
    #             for k = 1:num_grids
    #                 if (grids[k][cont][cont2][cont3])
    #                     if !(grids[k][cont][cont2][cont3-1])
    #                         OUTPUTgrids[k, 5][cont, cont2, cont3] = true
    #                     end
    #                     if !(grids[k][cont][cont2][cont3+1])
    #                         OUTPUTgrids[k, 6][cont, cont2, cont3] = true
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    return OUTPUTgrids
end
