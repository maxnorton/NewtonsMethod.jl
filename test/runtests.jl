using NewtonsMethod, LinearAlgebra
using Test

f(x)=-x^2; fp(x)=-2x
g(x)=2(x-4)^3; gp(x) = 6(x-4)^2
h(x)=x^3-2x^2-11x+12; hp(x)=3x^2-4x-11
x=1.0; x1=2.35287527; x2=2.35284172
tol=1E-6 # use manual tolerance for known-function tests

@testset "Fns with known roots" begin
    @test norm(newtonroot(f, fp, x)[1]) < tol;
    @test norm(newtonroot(g, gp, x)[1] - 4) < tol;
    #testing known basins of attraction for h
    @test norm(newtonroot(h, hp, x1)[1] - 4) < tol;
    @test norm(newtonroot(h, hp, x2)[1] + 3) < tol;
end

@testset "Autodiff known fns" begin
    @test norm(newtonroot(f, x)[1]) < tol;
    @test norm(newtonroot(g, x)[1] - 4) < tol;
    @test norm(newtonroot(h, x1)[1] - 4) < tol;
    @test norm(newtonroot(h, x2)[1] + 3) < tol;
end
