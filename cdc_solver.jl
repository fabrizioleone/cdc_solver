module cdc_solver

export solve_cdc

############################
## EXPOSED USER FUNCTIONS ##
############################
function solve_cdc(; N::Int, obj::Function, equalise_obj::Function, zero_D_j_obj::Function, max_iter::Int = 100)
    if N < 1
        error("number of choice elements must be positive")
    end
    if max_iter < 1
        error("maximum # of iterations must be positive")
    end

    # N: the number of on-off choices
    # obj(J, prod): the objective function
    # equalise_obj(J1, J2): finds phi so that obj(J1, phi) - obj(J2, phi) = 0
    # zero_D_j_obj(J, j): finds phi so that obj(J union j, phi) - obj(J without j, phi) = 0

    global user_N, user_obj, user_equalise_obj, user_zero_D_j_obj, user_max_iter
    user_N = N
    user_obj = obj
    user_equalise_obj = equalise_obj
    user_zero_D_j_obj = zero_D_j_obj
    user_max_iter = max_iter

    # initialise, then run algorithm
    none = falses(user_N)
    ## below left0: J_opt is empty
    left0 = minimum([user_zero_D_j_obj(none, j) for j in 1:user_N])
    ## above right0: J_opt is full
    right0 = maximum([user_zero_D_j_obj(.~ none, j) for j in 1:user_N])
    interval0 = prod_interval(left0, right0, none, .~ none)
    ## run algorithm for interior
    solved = solve_interval(interval0)

    # paste together adjacent intervals with identical decisions
    i_new_J = [solved[n-1].inf != solved[n].inf for n in 2:length(solved)]
    pushfirst!(i_new_J, true) # the first interval in solved should be included
    new_J = [int.inf for int in solved[i_new_J]]
    cutoffs = [int.left for int in solved[i_new_J]]
    push!(cutoffs, right0)

    # add in the (0, left0) and (right0, Inf) intervals
    pushfirst!(cutoffs, 0)
    pushfirst!(new_J, none)
    push!(cutoffs, Inf)
    push!(new_J, .~ none)

    (new_J, cutoffs)
end

######################
## TYPE DEFINITIONS ##
######################
# setting up interval structures
struct prod_interval
    left::Number
    right::Number
    inf::AbstractArray{Bool,1}
    sup::AbstractArray{Bool,1}

    function prod_interval(left, right, inf, sup)
        if left > right
            error("left and right do not define a valid interval")
        end
        if size(inf) != size(sup)
            error("inf and sup are not the same length")
        end
        if any(inf .> sup)
            error("inf is not contained in sup")
        end
        new(left, right, inf, sup)
    end
end
function isequal(int1::prod_interval, int2::prod_interval)
    (int1.left == int2.left) && (int1.right == int2.right) && (int1.inf == int2.inf) && (int1.sup == int2.sup)
end
function Base.isless(int1::prod_interval, int2::prod_interval)
    int1.right <= int2.left
end
function Base.in(prod::Number, interval::prod_interval)
    prod < interval.right && prod > interval.left
end
function Base.print(interval::prod_interval)
    print("interval: [", interval.left, ", ", interval.right, ") \n")
    return hcat(interval.inf, interval.sup)
end

#########################
## ALGORITHM FUNCTIONS ##
#########################
# inner loop: iterating algorithm until inf_n = inf_{n+1} and same for sup
## iterater
function converge_interval(interval0::prod_interval)
    # initialisation
    converged = prod_interval[]
    iterating = prod_interval[]
    push!(iterating, interval0)

    for n = 1:user_max_iter
        update = [update_interval(int) for int in iterating]
        done = [check_interval_converged(pair...) for pair in zip(iterating, update)]
        append!(converged, vcat(update[done]...))
        iterating = vcat(update[.~ done]...)
        if isempty(iterating)
            break
        end
        if n == user_max_iter
            error("interval did not converge to new intervals")
        end
    end

    sort!(converged)
end
function check_interval_converged(int::prod_interval, update::Array{prod_interval, 1})
    length(update) > 1 && return false
    (int.left == update[1].left) && (int.right == update[1].right) && (int.inf == update[1].inf) && (int.sup == update[1].sup)
end
## update for the iterator: takes an interval, returns an array of intervals due to update
function update_interval(interval0::prod_interval)
    maybes = findall(map(xor, interval0.inf, interval0.sup))

    # find the cutoffs for D_j(inf) = 0 (and sup) for each j in maybes
    # notes: these vectors are length N_maybes
    cutoffs_inf_all = [user_zero_D_j_obj(interval0.sup, j) for j in maybes]
    cutoffs_sup_all = [user_zero_D_j_obj(interval0.inf, j) for j in maybes]

    # sup-type cutoffs
    ## js with cutoffs to the right of the interval should be dropped
    base_sup = similar(interval0.sup)
    copyto!(base_sup, interval0.sup)
    base_sup[maybes[cutoffs_sup_all .>= interval0.right]] .= false
    ## js with cutoffs within the interval need to spawn new intervals
    i_cutoffs_sup = [in(cutoff, interval0) for cutoff in cutoffs_sup_all]

    # inf-type cutoffs
    ## js with cutoffs to the left of the interval should be turned on
    i_add = [cutoff <= interval0.left for cutoff in cutoffs_inf_all]
    base_inf = similar(interval0.inf)
    copyto!(base_inf, interval0.inf)
    base_inf[maybes[i_add]] .= true
    ## js with cutoffs within the interval need to spawn new intervals: length N_maybe
    i_cutoffs_inf = [in(cutoff, interval0) for cutoff in cutoffs_inf_all]

    i_cutoffs = vcat(i_cutoffs_inf, i_cutoffs_sup)
    # if no cutoffs in original interval, no update to interval
    if .!i_cutoffs |> all
        update = prod_interval[]
        push!(update, prod_interval(interval0.left, interval0.right, base_inf, base_sup))
        return update
    end

    # otherwise, need to update and return a tuple of the new intervals
    ## store the relevant cutoffs in M-length vector
    cutoffs = vcat(cutoffs_inf_all[i_cutoffs_inf], cutoffs_sup_all[i_cutoffs_sup])
    cutoffs_type = vcat(zeros(Int, sum(i_cutoffs_inf)), ones(Int, sum(i_cutoffs_sup)))
    cutoffs_j = vcat(maybes[i_cutoffs_inf], maybes[i_cutoffs_sup])
    M = length(cutoffs)
    ## sort according to cutoffs
    cutoffs_type = cutoffs_type[sortperm(cutoffs)]
    cutoffs_j = cutoffs_j[sortperm(cutoffs)]
    sort!(cutoffs)
    ## calculate the M-length vector of new infs
    new_infs_matrix = repeat(reshape(base_inf, user_N, 1), 1, M)
    ### for each m corresponding to a inf-type cutoffs, turn on j for m and all following intervals
    map(m -> new_infs_matrix[cutoffs_j[m], m:M] .= true, findall(cutoffs_type .== 0))
    new_infs = [reshape(new_infs_matrix[:, m], user_N) for m in 1:M]
    pushfirst!(new_infs, base_inf)
    ## same, for sups
    new_sups_matrix = repeat(reshape(base_sup, user_N, 1), 1, M)
    ### for each m corresponding to a sup-type cutoff, turn off j for m and all preceding intervals
    map(m -> new_sups_matrix[cutoffs_j[m], 1:m] .= false, findall(cutoffs_type .== 1))
    new_sups = [reshape(new_sups_matrix[:, m], user_N) for m in 1:M]
    push!(new_sups, base_sup)
    ## we should end up with M+1 intervals
    push!(cutoffs, interval0.right)
    pushfirst!(cutoffs, interval0.left)
    new_left = cutoffs[1:M+1]
    new_right = cutoffs[2:M+2]
    ## finally, exclude any measure-0 intervals
    i_not_measure0 = (new_left .!= new_right)

    # return new intervals
    [prod_interval(new_left[n], new_right[n], new_infs[n], new_sups[n]) for n in findall(i_not_measure0)]
end

# outer loop: update interval into set of intervals with inf=sup
## iterator
function converge_interval_infsups(interval0::prod_interval)
    solved = prod_interval[]
    update = converge_interval(interval0)
    # intervals with inf = sup are solved
    done = [sum(int.sup) - sum(int.inf) == 0 for int in update]
    append!(solved, update[done])

    # iterate other intervals until we have a set of inf=sup intervals
    iterating = prod_interval[]
    converged = prod_interval[]
    append!(iterating, update[.~ done])
    for n = 1:user_max_iter
        update = vcat([split_set_space(int) for int in iterating]...)
        done = [sum(int.sup) - sum(int.inf) == 0 for int in update]
        append!(converged, update[done])
        iterating = update[.~ done]
        if isempty(iterating)
            break
        end
        if n == user_max_iter
            error("interval did not converge to (inf=sup)-type intervals")
        end
    end
    sort!(converged)
    return(solved, converged)
end
## update for iterator: takes a converged interval, returns a set of intervals
function split_set_space(interval0::prod_interval)
    # pick the first j in maybe
    j = map(xor, interval0.sup, interval0.inf) |> findfirst

    # make interval with it definitely turned on turn it on
    new_inf = similar(interval0.inf)
    copyto!(new_inf, interval0.inf)
    new_inf[j] = true

    # make interval with it definitely turned off
    new_sup = similar(interval0.sup)
    copyto!(new_sup, interval0.sup)
    new_sup[j] = false

    interval_j_on = prod_interval(interval0.left, interval0.right, new_inf, interval0.sup)
    interval_j_off = prod_interval(interval0.left, interval0.right, interval0.inf, new_sup)

    # converge both intervals
    return vcat(converge_interval(interval_j_on), converge_interval(interval_j_off))
end

# final step: take intervals with multiple J=inf=sup options and compare them
function solve_interval(interval0::prod_interval)
    (solved, converged) = converge_interval_infsups(interval0)

    # calculate intervals with multiple inf=sup options
    ## new intervals
    new_endpoints = vcat([[int.left; int.right] for int in converged]...) |> unique
    sort!(new_endpoints)
    new_left = new_endpoints[1:length(new_endpoints)-1]
    new_right = new_endpoints[2:length(new_endpoints)]
    ## intervals from converged
    base_left = [int.left for int in converged]
    base_right = [int.right for int in converged]
    base_Js = [int.inf for int in converged]

    # find the inf=sup options for each interval
    new_Js = [base_Js[(new_left[int] .>= base_left) .& (new_right[int] .<= base_right)] for int in eachindex(new_left)]
    ## keep intervals with actual J options
    drop = [length(J_set) == 0 for J_set in new_Js]
    deleteat!(new_left, drop)
    deleteat!(new_right, drop)
    deleteat!(new_Js, drop)
    solved_converged = [calc_global_max(new_left[n], new_right[n], new_Js[n]) for n in 1:length(new_left)]

    append!(solved, vcat(solved_converged...))
    sort!(solved)
end
## take an interval with multiple possible Js, find the cutoffs for each as global maxima
function calc_global_max(left::Number, right::Number, Js::Array{<: AbstractArray{Bool}})
    # generate all pairs of Js
    J_pairs = Array{AbstractArray{Bool}}[]
    for J1 in 2:length(Js)
        for J2 in 1:J1-1
            push!(J_pairs, [Js[J1], Js[J2]])
        end
    end
    # create intervals based on when objs cross each other
    break_points = [user_equalise_obj(pair...) for pair in J_pairs] |> unique
    ## drop crossing points that aren't in the original interval
    drop = [(point < left) || (point > right) for point in break_points]
    deleteat!(break_points, drop)
    sort!(break_points)
    ## add original interval boundaries
    push!(break_points, right)
    pushfirst!(break_points, left)

    # within each interval, one J will be the best. just pick the midpoints and calculate obj for all J options to pick the global maximum
    midpoints = [0.5break_points[n] + 0.5break_points[n+1] for n = 1:length(break_points)-1]
    i_best = [argmax([user_obj(J, point) for J in Js]) for point in midpoints]

    return [prod_interval(break_points[n], break_points[n+1], Js[i_best[n]], Js[i_best[n]]) for n = 1:length(midpoints)]
end

end
