using Statistics

a = [1, 2, 3]
b = [2, 2, 2]'
a*b

# test creating a matrix from one vector

# test finding elements in array and changing them
a = Array(1:10)
indices = findall(a -> a >= 5, a)
a[indices] .= 5

a = Array(1:10)
indices = findall(a -> a >= 11, a)
a[indices] .= 5
a

mean(a*b, dims=1)
mean(a*b, dims=2)

plus_one(2)

function plus_one(n)
    return n+1
end