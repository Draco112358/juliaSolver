using Base.Sort: searchsortedfirst

# function bin_search(num::Float64, A::Array{Float64,1})
#     index = 0
#     n = length(A)
#     left = 1
#     right = n
#     while left <= right
#         mid = ceil(Int64, (left + right) / 2)
#         if A[mid] == num
#             index = mid
#             break
#         else
#             if A[mid] > num
#                 right = mid - 1
#             else
#                 left = mid + 1
#             end
#         end
#     end
#     return convert(Int64,index)
# end


function bin_search(num::Float64, A::Array{Float64,1})
    index = searchsortedfirst(A, num)
    if index > length(A) || A[index] != num
        index = 0
    end
    return convert(Int64,index)
end