function indiv_alg(th, rh, Gamma, t, f, ga_ji, d_jk, M, xi, pro_min, L, A, T; z::Float64, C::Int, c::Array{Float64, 1}, X::Array{Float64, 1}, P_exp::Array{Float64, 1}, i::Int, max_iter::Int = 100)
    Theta_blocks_jki = @. (1 - t)^(th/(rh-1)) * (c*ga_ji*d_jk/T).^(-th)
    # k-country demand: X*P^(rh-1)
    B_k = reshape(X./P_exp, 1, C)
    Theta_k(J::AbstractArray{Bool, 1}) = sum(Theta_blocks_jki[:, :, i] .* J, dims = 1)
    Pi(J) = rh*A*z*sum(B_k.*Theta_k(J).^((rh-1)/th)) - sum(c.*f.*(1 .- t).*J)
    # Pi(J) = obj(J, z)
    function D_j_Pi(J::BitArray{1}; j::Int)
        J_j_on = similar(J)
        J_j_off = similar(J)
        copyto!(J_j_on, J)
        copyto!(J_j_off, J)

        J_j_on[j] = true;
        J_j_off[j] = false;

        Pi(J_j_on) - Pi(J_j_off)
    end
    function update_pair(; inf::BitArray{1}, sup::BitArray{1})
        max_mv = [D_j_Pi(inf; j = j) for j in 1:C]
        min_mv = [D_j_Pi(sup; j = j) for j in 1:C]

        new_inf = (inf .| (min_mv .> 0)) .& sup
        new_sup = (max_mv .> 0) .& sup

        (new_inf, new_sup)
    end
    function converge_sandwich(; inf0::BitArray{1}, sup0::BitArray{1}, max_iter = 100)
        inf = similar(inf0)
        sup = similar(sup0)
        copyto!(inf, inf0)
        copyto!(sup, sup0)

        converged = false
        for n in 1:max_iter
            (new_inf, new_sup) = update_pair(inf = inf, sup = sup)
            if vcat(new_inf, new_sup) == vcat(inf, sup)
                converged = true
                break
            end
            inf = new_inf
            sup = new_sup
        end
        if ~converged
            error("no sandwich convergence")
        end
        return (inf, sup)
    end
    function split_eval(inf, sup)
        j = findfirst(sup .& (.~ inf))
        infj1 = similar(inf)
        copyto!(infj1, inf)
        infj1[j] = true
        supj0 = similar(sup)
        copyto!(supj0, sup)
        supj0[j] = false

        (infj1, supj1) = converge_sandwich(inf0 = infj1, sup0 = sup)
        (infj0, supj0) = converge_sandwich(inf0 = inf, sup0 = supj0)

        return [[infj1, supj1], [infj0, supj0]]
    end
    function solve()
        (inf, sup) = converge_sandwich(inf0 = falses(C), sup0 = trues(C))
        if inf == sup
            return inf
        end

        possibilities = Array{Array{BitArray{1}, 1}}(undef, 0)
        iterating = Array{Array{BitArray{1}, 1}}(undef, 0)
        push!(iterating, [inf, sup])
        converged = false

        for n in 1:max_iter
            iterating = vcat([split_eval(pair...) for pair in iterating]...)
            done = [reduce(==, pair) for pair in iterating]
            append!(possibilities, iterating[done])
            deleteat!(iterating, done)
            if isempty(iterating)
                converged = true
                break
            end
        end
        if !converged
            error("no converge on set space splitting")
        end

        possibilities = [pair[1] for pair in possibilities]
        i_J_opt = findmax(Pi.(possibilities))
    end

    solve()
end
