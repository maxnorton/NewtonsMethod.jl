module NewtonsMethod

using LinearAlgebra, Statistics, Compat, ForwardDiff

function newtonroot(f, f′, x₀; tol=1E-7, maxiter=1000)
    x_old = x₀
    normdiff = Inf
    iter = 1
    while normdiff >= tol && iter <= maxiter
        x_new = x_old - f(x_old)/f′(x_old)
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter += 1
    end
    normdiff > tol ? (nothing, normdiff, iter) : (x_old, normdiff, iter)
end

function newtonroot(f, x₀; tol=1E-7, maxiter=1000)
    Diff(f) = x -> ForwardDiff.derivative(f, x)
    f′ = Diff(f)
    newtonroot(f, f′, x₀, tol=tol, maxiter=maxiter)
end

export newtonroot

end
