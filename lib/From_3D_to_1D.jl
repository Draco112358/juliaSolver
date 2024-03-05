function From_3D_to_1D(i, j, k, M, N) 
	pos = ((k-1) * M * N) + ((j-1) * M) + i;
    return convert(Int64,pos)
end