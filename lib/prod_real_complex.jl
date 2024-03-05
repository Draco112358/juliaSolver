using SparseArrays

function prod_real_complex(A::SparseMatrixCSC,x)
    return A*x
end


function prod_real_transposed_complex(A::SparseMatrixCSC, x)
    return (x'*A)'
end