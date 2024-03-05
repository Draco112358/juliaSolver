function crossbar_250_structure_lumped_elements_definition()
    # define the lumped elements position in meters
    num_le = 7
    le_end = zeros(num_le, 3)
    le_start = zeros(num_le, 3)
    lumped_elements = Dict("value" => [1e-6*ones(5,1);50*ones(2,1)], "type" => ones(num_le,1))
    coord_xy = [0.0750, 0.175, 0.275, 0.375, 0.475]
    k = 0
    for c1 = 1:5
        c2 = c1
#         for c2=1:5
            k = k + 1
            le_start[k,:] = [coord_xy[c1], coord_xy[c2], 0.0125]*1e-3
            le_end[k,:] = [coord_xy[c1], coord_xy[c2], 0.025]*1e-3
#         end
    end
    k = k + 1
    le_start[k,:] = [0, 0.375, 0.0312]*1e-3
    le_end[k,:] = [0, 0.475, 0.0312]*1e-3
    k = k + 1
    le_start[k,:] = [0.375, 0, 0.0063]*1e-3
    le_end[k,:] = [0.475, 0, 0.0063]*1e-3
    lumped_elements["le_start"] = le_start
    lumped_elements["le_end"] = le_end
    lumped_elements["s_le_start"] = []
    lumped_elements["s_le_end"] = []
    return lumped_elements
end
