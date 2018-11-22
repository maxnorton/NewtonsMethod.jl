using NewtonsMethod, LinearAlgebra
using Test

f(x)=-x^2; fp(x)=-2x
g(x)=2(x-4)^3; gp(x) = 6(x-4)^2
h(x)=x^3-2x^2-11x+12; hp(x)=3x^2-4x-11
n(x)=x^2+42.0; np(x)=2x
x₀=1.0; xb₀=one(BigFloat); x1=2.35287527; x2=2.35284172
tol=1E-6 # use manual tolerance for known-function tests

for x in [x₀, xb₀]
    @testset "Fns with known roots" begin
        @test norm(newtonroot(f, fp, x)[1]) < tol
        @test norm(newtonroot(g, gp, x)[1] - 4) < tol
        #known basins of attraction for h
        @test norm(newtonroot(h, hp, x1)[1] - 4) < tol
        @test norm(newtonroot(h, hp, x2)[1] + 3) < tol
        #nonconvergence
        @test newtonroot(n, np, x)[1]==nothing
    end

    @testset "Autodiff known fns" begin
        @test norm(newtonroot(f, x)[1]) < tol
        @test norm(newtonroot(g, x)[1] - 4.0) < tol
        @test norm(newtonroot(h, x1)[1] - 4.0) < tol
        @test norm(newtonroot(h, x2)[1] + 3.0) < tol
        @test newtonroot(n, x)[1]==nothing # redundant?
        @test newtonroot(n, x)==newtonroot(n, np, x) # test of f′=Diff(f)
    end
end

@testset "Maxiter, iter, tol" begin
    # expected nonconvergence behavior for complicated function and small maxiter
    @test newtonroot(h, hp, x1, maxiter=5)[1]==nothing
    @test newtonroot(h, x1, maxiter=5)[1]==nothing
    # OK to set tol as exactly 0 when NM yields exact root
    @test newtonroot(h, hp, x1, tol=0.0)[1] == 4.0
    @test newtonroot(h, x1, tol=0.0)[1] == 4.0
    # NM cannot get exact root, so tol=0 returns nothing when x has arbitrary precision
    @test newtonroot(f, fp, one(BigFloat), tol=0.0)[1] == nothing
    @test newtonroot(f, one(BigFloat), tol=0.0)[1] == nothing
    # when tol is nonzero, tol' > tol yields less precise estimate
    @test norm(newtonroot(f, fp, x₀, tol=1E-4)[1] - 0) > norm(newtonroot(f, fp, x₀, tol=1E-5)[1] - 0)
    # tests for iter. nonideal logic, i think.
    @test newtonroot(h, hp, x1, maxiter=0)[3]==1
    @test newtonroot(h, hp, x1, tol=Inf)[3]==1
end
