using LinearAlgebra
function crossbar_250_structure_nodes_ports_definition()
    # import necessary libraries
    
    # define the port position in meters
    num_ports = 2
    port_start = zeros(num_ports, 3)
    port_end = zeros(num_ports, 3)
    k = 1
    port_start[k, :] = [0, 0.375, 0.0312] .* 1e-3
    port_end[k, :] = [0, 0.475, 0.0312] .* 1e-3
    k += 1
    port_start[k, :] = [0.375, 0, 0.0063] .* 1e-3
    port_end[k, :] = [0.475, 0, 0.0063] .* 1e-3
    k += 1
    nodes_ports = Dict("port_start" => port_start, "port_end" => port_end, "s_port_start" => zeros(0, 6), "s_port_end" => zeros(0, 6))
    return nodes_ports
end