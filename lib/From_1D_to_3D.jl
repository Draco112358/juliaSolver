function From_1D_to_3D( M, N, pos)
    pos=pos-1;
	k = floor(pos / (M * N));
	j = floor((pos - k* M * N) / M);
	i = mod(pos - k * M * N, M);
    k=k+1;
    j=j+1;
    i=i+1;
    return convert(Int64,i),convert(Int64,j),convert(Int64,k)
end