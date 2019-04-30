using LinearAlgebra
import SpecialFunctions
import Random
import IterTools
Random.seed!(1234)

include("cdc_solver.jl")
using Main.cdc_solver
include("individual_algorithm.jl")

function gen_params(; C::Int = 15)
    th = 5
    rh = 4
    Gamma = SpecialFunctions.gamma((th-rh+1)/th)
    t = rand(Float64, C)/5
    f = fill(3, C)
    ga_ji = rand(Float64, C, C) .+ 1
    ga_ji[diagind(ga_ji)] .= 1
    ga_ji = reshape(ga_ji', C, 1, C)
    d_jk = rand(Float64, C, C) .+ 1
    d_jk[diagind(d_jk)] .= 1
    M = rand(Float64, C) .+ 1
    xi = fill(6, C)
    pro_min = rand(Float64, C) .+ 1
    L = rand(Float64, C) .+ 1
    T = fill(Gamma^(1/(1-rh)), C)

    A = (1/rh) * (rh/(rh-1))^(1-rh) * Gamma

    return th, rh, Gamma, t, f, ga_ji, d_jk, M, xi, pro_min, L, A, T
end
function solve_simple_eq(th, rh, Gamma, t, f, ga_ji, d_jk, M, xi, pro_min, L, A, T; C::Int, tol = 1e-9, max_iter = 100)
    # pre-calculations
    ## (th_min/th_cut)^xi
    ratio_exp = @. L/(M*f*(1 + (rh-1)*(xi/(xi-1))))
    ## cutoffs
    th_cut = @. pro_min*ratio_exp^(-1/xi)
    ## own gammas
    ga_own = ga_ji[1:C+1:C*C]

    function update_agg(c::Array{<:Number, 1})
        weight_nums = @. (c*ga_own/T)^(1-rh)*th_cut*ratio_exp*M*(xi/(xi-1))
        weight_denoms_k = @. $sum(weight_nums * (d_jk)^(1-rh), dims = 1)
        weights_k = reshape(weight_nums, 1, C)./weight_denoms_k

        eigenvec_k = @. $reshape(c*f*(c*ga_own/T)^(rh-1)/th_cut, 1, C)
        eigenvec_update = @. $reshape($sum(eigenvec_k * (d_jk)^(1-rh) * weights_k, dims = 2), C)

        c_new = @. (eigenvec_update*th_cut*(ga_own/T)^(1-rh)/f)^(1/rh)
        c_new = c_new/last(c_new)
    end

    c = ones(Int, C)
    for n in 1:max_iter
        c_new = update_agg(c)
        diff = @. $sum(abs((c - c_new)/c))
        # print("simple equilibrium diff: ", diff, "\n")
        if diff < tol
            break
        end
        if n == max_iter
            error("no convergence")
        end
        c = 0.5c_new + 0.5c
    end
    # return c

    # calculate X and P_exp
    P_exp = @. rh*A*$sum((c*ga_own*d_jk/T)^(1-rh)*th_cut*ratio_exp*M*(xi/(xi-1)), dims = 1)
    P_exp = reshape(P_exp, C)
    X =  @. rh*(xi/(xi-1))*ratio_exp*c*f*M
    return X, P_exp, c
end
function eval_alg_for_params(th, rh, Gamma, t, f, ga_ji, d_jk, M, xi, pro_min, L, A, T; C::Int, c::Array{Float64, 1}, X::Array{Float64, 1}, P_exp::Array{Float64, 1}, i::Int)
    Theta_blocks_jki = @. (1 - t)^(th/(rh-1)) * (c*ga_ji*d_jk/T).^(-th)
    # k-country demand: X*P^(rh-1)
    B_k = reshape(X./P_exp, 1, C)
    Theta_k(J::AbstractArray{Bool, 1}; i::Int) = sum(Theta_blocks_jki[:, :, i] .* J, dims = 1)
    function solve_i(; c::Array{Float64, 1}, i::Int, Theta_blocks_jki::Array{Float64, 3}, B_k::Array{Float64, 2}, Theta_k::Function)
        #############################
        ## set up for firm problem ##
        #############################
        function cdc_alg_i(; i::Int)
            function Pi(J::AbstractArray{Bool, 1}, pro)
                profit = @. A*pro*$sum(B_k * $Theta_k(J, i = i)^((rh-1)/th), dims = 2) - $sum(J*c*f*(1-t))
                return first(profit) # return as scalar
            end
            function equalise_Pi(J1::AbstractArray{Bool, 1}, J2::AbstractArray{Bool, 1})
                if size(J1) != size(J2)
                    error("J1 and J2 not the same length")
                end
                pro = @. $sum((J1 - J2)*c*f*(1-t), dims = 1)/(A*$sum(B_k*($Theta_k(J1, i = i)^((rh-1)/th) - $Theta_k(J2, i = i)^((rh-1)/th) ), dims = 2) )
                return first(pro) # return as scalar
            end
            function zero_D_j_Pi(J::AbstractArray{Bool,1}, j::Int)
                J_j_on = similar(J)
                J_j_off = similar(J)
                copyto!(J_j_on, J)
                copyto!(J_j_off, J)
                J_j_on[j] = true
                J_j_off[j] = false

                pro = @. c[j]*f[j]*(1-t[j])/(A*$sum(B_k*($Theta_k(J_j_on, i = i)^((rh-1)/th) - $Theta_k(J_j_off, i = i)^((rh-1)/th)), dims = 2) )
                return first(pro)
            end
            cdc_solver.solve_cdc(N = C, obj = Pi, equalise_obj = equalise_Pi, zero_D_j_obj = zero_D_j_Pi)
        end
        ########################
        ## solve firm problem ##
        ########################
        (J, cutoffs) = cdc_alg_i(i = i)
        # modify cutoffs to be above pro_min[i]
        ins = sum(cutoffs .<= pro_min[i])
        splice!(cutoffs, 1:ins, pro_min[i])
        deleteat!(J, 1:length(J)-length(cutoffs)+1)
        return (cutoffs, J)
    end

    solve_i(i = i, c = c, B_k = B_k, Theta_blocks_jki = Theta_blocks_jki, Theta_k = Theta_k)
end

#### TIME TRIALS ####
function run_alg_trials(trials::Int = 100; C::Int = 15)
    params = gen_params(C = C)
    times_alg = Array{Float64, 2}(undef, C, trials+1)
    times_ind = similar(times_alg)

    for n in 1:trials+1
        # print(n,": ")
        (X, P_exp, c) = solve_simple_eq(params..., C = C)

        # my algorithm
        times_alg[:, n] = [@elapsed eval_alg_for_params(params..., C = C, c = c, P_exp = P_exp, X = X, i = i) for i in 1:C]
        # print(sum(times_alg[:, n]),"\n")

        # individual algorithm
        z = 1:0.1:10
        times_ind[:, n] = [@elapsed broadcast(z -> indiv_alg(params...; C = C, c = c, X = X, P_exp = P_exp, i = i, z = z), z) for i in 1:C]

        params = gen_params(C = C)
    end
    (times_alg, times_ind)
end
# naive solution
function naive_soln(; C::Int = 15, z::Float64 = 2., i::Int = 1)
    params = gen_params(C = C)
    (th, rh, Gamma, t, f, ga_ji, d_jk, M, xi, pro_min, L, A, T) = params
    (X, P_exp, c) = solve_simple_eq(params..., C = C)

    Theta_blocks_jki = @. (1 - t)^(th/(rh-1)) * (c*ga_ji*d_jk/T).^(-th)
    # k-country demand: X*P^(rh-1)
    B_k = reshape(X./P_exp, 1, C)
    Theta_k(J::Array{Int, 1}) = sum(Theta_blocks_jki[J, :, i], dims = 1)
    Pi(J) = rh*A*z*sum(B_k.*Theta_k(J).^((rh-1)/th)) - sum(c[J].*f[J].*(1 .- t[J]))

    @elapsed begin
        Js = collect(IterTools.subsets(1:C))
        Pis = Array{Float64, 1}(undef, length(Js))

        for i_J in eachindex(Js)
            Pis[i_J] = Pi(Js[i_J])
        end
        findmax(Pis)
    end
end
