using Test, AdvancedBoosting

@testset "IdentityTransform" begin
    transform = IdentityTransform([1])

    @test transform([1.0]) == [1.0]
    @test transform([1.0, 2.0]) == [1.0]

    @test transform([1.0], [1.0]) == [1.0]
    @test transform([1.0, 2.0], [1.0, 2.0]) == [1.0]

    transform2 = IdentityTransform([1, 2])
    @test transform2([1.0, 2.0]) == [1.0, 2.0]

    @test transform2([1.0, 2.0], [1.0, 2.0]) == [1.0, 2.0]
end

@testset "SoftplusTransform" begin
    transform = SoftplusTransform([1])

    sp(x) = log(exp(x) + 1.0)

    @test transform([1.0]) == sp.([1.0])
    @test transform([1.0, 2.0]) == sp.([1.0])

    @test transform([1.0], [1.0]) == sp.([1.0])
    @test transform([1.0, 2.0], [1.0, 2.0]) == sp.([1.0])

    transform2 = SoftplusTransform([1, 2])
    @test transform2([1.0, 2.0]) == sp.([1.0, 2.0])

    @test transform2([1.0, 2.0], [1.0, 2.0]) == sp.([1.0, 2.0])
end

@testset "SigmoidTransform" begin
    transform = SigmoidTransform([1])

    sig(x) = 1.0 / (1.0 + exp(-x))

    @test transform([1.0]) == sig.([1.0])
    @test transform([1.0, 2.0]) == sig.([1.0])

    @test transform([1.0], [1.0]) == sig.([1.0])
    @test transform([1.0, 2.0], [1.0, 2.0]) == sig.([1.0])

    transform2 = SigmoidTransform([1, 2])
    @test transform2([1.0, 2.0]) == sig.([1.0, 2.0])

    @test transform2([1.0, 2.0], [1.0, 2.0]) == sig.([1.0, 2.0])
end

@testset "MultiTransform" begin
    transform = MultiTransform([IdentityTransform([1]), SigmoidTransform([1])])

    sig(x) = 1.0 / (1.0 + exp(-x))

    @test transform([1.0]) == [1.0, sig(1.0)]
    @test transform([1.0, 2.0]) == [1.0, sig(1.0)]

    @test transform([1.0], [1.0]) == [1.0, sig(1.0)]
    @test transform([1.0, 2.0], [1.0, 2.0]) == [1.0, sig(1.0)]

    transform2 = MultiTransform([IdentityTransform([1, 2]), SigmoidTransform([1, 2])])
    @test transform2([1.0, 2.0]) == [1.0, 2.0, sig(1.0), sig(2.0)]

    @test transform2([1.0, 2.0], [1.0, 2.0]) == [1.0, 2.0, sig(1.0), sig(2.0)]
end

@testset "VaryingCoefficientTransform" begin
    transform = VaryingCoefficientTransform([1])
    @test transform([1.0], [1.0]) == [2.0]

    transform2 = VaryingCoefficientTransform([1,2])
    @test transform2([1.0, 2.0], [1.0, 2.0]) == [6.0]
end

@testset "ComposedTransform" begin
    g1 = VaryingCoefficientTransform([1])
    g2 = SoftplusTransform([1])

    transform = g2 ∘ g1
    @test transform([1.0], [1.0]) == [log(exp(2.0)+1.0)]

    g1_2 = VaryingCoefficientTransform([1,2])
    transform2 = g2 ∘ g1_2
    @test transform2([1.0, 2.0], [1.0, 2.0]) == [log(exp(6.0)+1.0)]
end

@testset "SumTransform" begin
    g1 = SoftplusTransform([1])
    g2 = SoftplusTransform([1])

    transform = g1 + g2
    @test transform([1.0],[1.0]) == [2 * log(exp(1.0)+1)]

    g1 = SoftplusTransform([1,2])
    g2 = SoftplusTransform([1,2])

    transform = g1 + g2
    @test transform([1.0,2.0],[1.0,2.0]) == 2 .* [log(exp(1.0)+1), log(exp(2.0)+1)]
end
