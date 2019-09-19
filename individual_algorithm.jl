function AE_alg(Pi::Function; N::Int, max_iter::Int = 100)
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
        max_mv = [D_j_Pi(inf; j = j) for j in 1:N]
        min_mv = [D_j_Pi(sup; j = j) for j in 1:N]

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
        (inf, sup) = converge_sandwich(inf0 = falses(N), sup0 = trues(N))
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
