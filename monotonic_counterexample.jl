module monotonic_counterexample
    import SpecialFunctions

    n = Int[9, 16]
    f = Rational[1, 5//3]
    N = 2;
    A = SpecialFunctions.gamma(5/6)*(1/4)*(4/3)^(-3)

    # calculate left0, so we can scale by it
    th0 = minimum(f ./ (2*A*n.^(1/2)))
    function obj(J::AbstractArray{Bool,1}, k::Number)
        (k*th0)*A*2*sum(J.*n).^(1/2) - sum(J .* f)
    end
    function zero_D_j_obj(J::AbstractArray{Bool,1}, j::Int)
        J_j_on = similar(J)
        J_j_off = similar(J)
        copyto!(J_j_on, J)
        copyto!(J_j_off, J)
        J_j_on[j] = true
        J_j_off[j] = false

        f[j]/(2*A*th0*( sum(J_j_on.*n)^(1/2) - sum(J_j_off.*n)^(1/2) ))
    end
    function equalise_obj(J1::AbstractArray{Bool,1}, J2::AbstractArray{Bool,1})
        (sum(J1 .* f) - sum(J2 .* f))/(2*A*th0*sum(J1 .* n)^(1/2) - 2*A*th0*sum(J2 .* n)^(1/2))
    end

    export N, obj, zero_D_j_obj, equalise_obj
end
