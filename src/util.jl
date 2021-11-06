# convenience functions to get generic Inf and NaN
inf(::Union{Type{T}, T}) where {T} = convert(T, Inf)
two(::Union{Type{T}, T}) where {T} = convert(T, 2)

"""
	dropped(f, A; dims)
Applies a reducing operator `f` (e.g. `sum`) on `A` over `dims`, and drops them.
"""
function dropped(f, A::AbstractArray; dims)
	S = f(A; dims=dims)
	return dropdims(S; dims=dims)
end

"""
    droppedfirst(f, A, Val(N))
Applies a reducing operator `f` (e.g. `sum`) on `A` over its first `N`
dimensions, and drops them.
"""
function droppedfirst(f, A::AbstractArray, ::Val{N}) where {N}
	dims = OneHot.tuplen(Val(N))
	return dropped(f, A; dims=dims)
end

"""
    sum_(A, dims)
Sums `A` over dimensions `dims` and drops them.
"""
sum_(A::AbstractArray; dims) = dropped(sum, A; dims=dims)

"""
    sumdropfirst(A, Val(N))
Sums `A` over its first `N` dimensions and drops them.
"""
sumdropfirst(A::AbstractArray, ::Val{N}) where {N} = droppedfirst(sum, A, Val(N))

"""
	diffdrop(A, dims)
Takes the difference of `A` across dimensions `dims` and drops them.
"""
diffdrop(A::AbstractArray; dims) = dropped(diff, A; dims=dims)

"""
	meandrop(A, dims)
Takes the mean of `A` across dimensions `dims` and drops them.
"""
meandrop(A::AbstractArray; dims) = dropped(mean, A; dims=dims)
stddrop(A::AbstractArray; dims) = dropped(std, A; dims=dims)

"""
	logsumexp(X; dims=:)
Like logsumexp from StatsFuns, but with a `dims` keyword.
"""
logsumexp(X::AbstractArray{<:Real}; dims=:) = _logsumexp(X, dims)
logsumexp(x::Number; dims::Colon=:) = x
# need to special-case dims = :
# See https://github.com/JuliaLang/julia/issues/28866#issuecomment-630285680.
function _logsumexp(X::AbstractArray{<:Real}, ::Colon)
    u = maximum(X)
	u .+ log.(sum(exp.(X .- u)))
end
function _logsumexp(X::AbstractArray{<:Real}, dims)
    u = maximum(X; dims=dims)
	u .+ log.(sum(exp.(X .- u); dims=dims))
end

"""
	logsumexpdrop(X; dims)
Like logsumexp, dropping resulting singleton `dims` dimensions.
"""
logsumexpdrop(X::AbstractArray; dims) = dropped(logsumexp, X; dims=dims)

@inline tail2(t::Tuple{Any,Any,Vararg{Any}}) = tail(tail(t))
@inline front2(t::Tuple{Any,Any,Vararg{Any}}) = front(front(t))

allequal(a, b, c...) = a == b && allequal(b, c...)
allequal(a) = true
allequal() = true

oddargs() = ()
oddargs(a, b, c...) = (a, oddargs(c...)...)
odditems(t::Tuple) = oddargs(t...)

evenargs() = ()
evenargs(a, b, c...) = (b, evenargs(c...)...)
evenitems(t::Tuple) = evenargs(t...)

interleave(x, t::Tuple{Any, Vararg{Any}}) = (x, first(t), interleave(x, tail(t))...)
interleave(x, t::Tuple{}) = (x,)

"""
	erfx(x)
Returns erf(x)/x, properly defined at x=0.
"""
function erfx(x)
	result = erf(x) / x
	return ifelse(iszero(x), oftype(result, 2/√π), result)
end

"""
	tuplejoin(tuples...)
Concatenates the argument tuples into a single tuple.
"""
@inline tuplejoin() = ()
@inline tuplejoin(x::Tuple, y::Tuple...) = (x..., tuplejoin(y...)...)

"""
	tuplesub(tuple, Val(i0), Val(i1))
Returns tuple[i0:i1], ensuring type-stability.
"""
@generated tuplesub(t::Tuple, ::Val{i}, ::Val{j}) where {i,j} =
    Expr(:tuple, (:(t[$k]) for k ∈ i:j)...)

"""
	tuplefill(v, Val(N))
Returns a tuple of the value `v` repeated `N` times.
"""
@generated tuplefill(v, ::Val{N}) where {N} = Expr(:tuple, (:v for i in 1:N)...)

"""
	staticgetindex(A, Val(I))
Returns A[I] as a tuple, ensuring type-stability.
"""
@generated staticgetindex(A, ::Val{I}) where {I} =
    Expr(:tuple, (:(A[$i]) for i ∈ I)...)

"""
	sizes(A, dims)
Sizes of `A` along given dimensions.
"""
@generated sizes(A::AbstractArray, dims::NTuple{N,Any}) where {N} =
	Expr(:tuple, (:(size(A, dims[$i])) for i ∈ 1:N)...)

"""
	staticsizes(A, Val(dims))
Static, sizes of `A` along given dimensions.
"""
@generated staticsizes(A::AbstractArray, ::Val{dims}) where {dims} =
	Expr(:tuple, (:(size(A, $d)) for d ∈ dims)...)

"""
	first2(x)
Returns first(first(x)).
"""
first2(x) = first(first(x))

"""
	select(cond, x, y)
Given a Boolean array `cond`, constructs an array with items from `x` or `y`
according to whether the corresinding item from `cond` is `true` or `false`.
"""
select(cond, x, y) = ifelse.(cond, x, y), ifelse.(cond, y, x)

"""
	weighted_mean(v, w)
Mean of `v` with weights `w`.
"""
function weighted_mean(v::AbstractArray, w::AbstractArray; dims=:)
    return mean(w .* v; dims=dims) / mean(w; dims=dims)
end
function weighted_mean(v::AbstractArray, w::Real; dims=:)
    return mean(v) # faster special case
end

"""
	Δ2(x, y)
Computes x^2 - y^2, using (x - y) * (x + y) which is numerically more accurate.
"""
Δ2(x, y) = (x - y) * (x + y)

# http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
function logsumexp_stream(X)
    alpha = -Inf
    r = 0.0
    for x in X
        if x ≤ alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    log(r) + alpha
end

# used by unzip
struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]
"""
	unzip(tuples)
Converts a vector of tuples into a tuple of vectors.
# Examples
```julia-repl
julia> unzip([(1,2,3), (5,6,7)])
([1, 5], [2, 6], [3, 7])
```
"""
@generated _unzip(tuples, ::Val{N}) where {N} =
 	Expr(:tuple, (:(map($(StaticGetter{i}()), tuples)) for i ∈ 1:N)...)
unzip(tuples) = _unzip(tuples, Val(length(first(tuples))))

"""
	throttlen(f, n)
Helper function to only run `f` every `n` times that it is called.
Note that the returned function always returns `nothing`, so the return
value of `f` is discarded.
"""
function throttlen(f, n)
    i = 1
    return function(args...; kwargs...)
        if i == n
            i = 1
            f(args...; kwargs...)
        else
            i += 1
        end
		return nothing # for type-stability
    end
end

function sigmoid(x::Real)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(one(t) + t), t / (one(t) + t))
end

"""
    generate_sequences(n, A = 0:1)
Retruns an iterator over all sequences of length `n` out of the alphabet `A`.
"""
function generate_sequences(n::Int, A = 0:1)
	return (collect(seq) for seq in Iterators.product(Iterators.repeated(A, n)...))
end
