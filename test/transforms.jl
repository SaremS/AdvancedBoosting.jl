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
    transform = VaryingCoefficientTransform()

    @test transform([1.0], [1.0]) == 2.0
    @test transform([1.0, 2.0], [1.0, 2.0]) == 6.0
end
