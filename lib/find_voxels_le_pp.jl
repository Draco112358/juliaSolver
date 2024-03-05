include("distfcm.jl")
include("bin_search.jl")

function find_voxels_le_pp(nodi_centri, lumped_elements, nodes, nodes_red)
    N = size(lumped_elements["le_start"], 1)
    lumped_elements["le_voxels"] = zeros(N, 2)
    lumped_elements["le_nodes"] = zeros(N, 2)
    for cont = 1:N
        lumped_elements["le_voxels"][cont, 1] = nodes_find_rev(lumped_elements["le_start"][cont, :], nodi_centri, -1)
        lumped_elements["le_voxels"][cont, 2] = nodes_find_rev(lumped_elements["le_end"][cont, :], nodi_centri, lumped_elements["le_voxels"][cont, 1])
        lumped_elements["le_nodes"][cont, 1] = bin_search(nodes[convert(Int64,lumped_elements["le_voxels"][cont, 1])], nodes_red)
        lumped_elements["le_nodes"][cont, 2] = bin_search(nodes[convert(Int64,lumped_elements["le_voxels"][cont, 2])], nodes_red)
    end
    return lumped_elements
end

function nodes_find_rev(Nodes_inp_coord, nodi_centri, node_to_skip)
    indici = sortperm(vec(distfcm(Nodes_inp_coord, nodi_centri)))
    if indici[1] != node_to_skip
        nodes = indici[1]
    else
        nodes = indici[2]
    end
    return nodes
end
