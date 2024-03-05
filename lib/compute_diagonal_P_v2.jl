include("compute_row_P_sup.jl")

function compute_diagonal_P_v2(N1,N2,N3,escalings,sx,sy,sz)
    self_P=zeros(3,1)
    centro_vox=[0 0 0]
    row_P=escalings["P"]*compute_row_P_sup(centro_vox,centro_vox,sx,sy,sz,1,1)
    self_P[1]=row_P[1]
    row_P=escalings["P"]*compute_row_P_sup(centro_vox,centro_vox,sx,sy,sz,3,3)
    self_P[2]=row_P[1]
    row_P=escalings["P"]*compute_row_P_sup(centro_vox,centro_vox,sx,sy,sz,5,5)
    self_P[3]=row_P[1]
    diag_P=vcat(
        fill(self_P[1], N1),
        fill(self_P[2], N2),
        fill(self_P[3], N3)
    )
    return diag_P
end
