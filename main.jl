using StatsBase, Random, DecisionTree

mutable struct _AAD
    seq::String
    val::Float64
end

function _AAD(seq::String)
    return _AAD(seq, 0)
end

function find_maxsequence(regr::RandomForestRegressor, s::AbstractRF, ami_arr::Int; blosum::Int=62, pool_rate::Float64=0.5, pool_iter::Int=4)
    _find_maxsequence(regr, s.fasta_loc, ami_arr, blosum, pool_rate, pool_iter)
end

function find_maxsequence(regr::RandomForestRegressor, fasta_loc::String, ami_arr::Int; blosum::Int=62, pool_rate::Float64=0.5, pool_iter::Int=4)
    _find_maxsequence(regr, fasta_loc, ami_arr, blosum, pool_rate, pool_iter)
end

function _random_gene_indata!(v::Vector{_AAD}, key_vector::Vector{Vector{Char}}; mutation::Float64=0.01)
    for aad in v
        seq_vector = collect(aad.seq)
        for ((ind, key)) in randsubseq(collect(enumerate(key_vector)), mutation)
            seq_vector[ind] = sample(key)
        end
        aad.seq = join(seq_vector)
    end
end

function _find_maxsequence(regr::RandomForestRegressor, fasta_loc::String, ami_arr::Int, blosum::Int, pool_rate::Float64, pool_iter::Int)
    data_len, loc_dict_vector, seq_matrix = _location_data(fasta_loc)
    blo = blosum_matrix(blosum)
    dict_key_vector = Vector{Vector{Char}}()
    x_col_vector = Vector{Vector{Float64}}()
    loc = Vector{Tuple{Int, Char}}()
    for (ind, (dict, col)) in enumerate(zip(loc_dict_vector, eachcol(seq_matrix)))
        max_val = maximum(values(dict))
        max_amino = _find_key(dict, max_val)
        dict_key = keys(dict)

        if '-' ∉ dict_key && ami_arr ≤ data_len - max_val 
            push!(x_col_vector, [blo[max_amino, i] for i in col])
            push!(loc, (ind, max_amino))
            push!(dict_key_vector, collect(dict_key))
        end
    end

    data_ind_vector = findall(x -> x ≤ data_len * pool_rate, sortperm(predict_data(regr, hcat(x_col_vector...)), rev=true))
    data_vector = Vector{_AAD}()
    for _ = 1:pool_iter
        for ind in data_ind_vector
            push!(data_vector, _AAD(join(seq_matrix[ind, :][[i[1] for i in loc]])))
        end
    end

    for _ = 1:1000
        Threads.@threads for aad in data_vector
            x_data_vector = [Float64(blo[am, tar]) for (am, (_, tar)) in zip(aad.seq, loc)]
            aad.val = DecisionTree.predict(regr, x_data_vector)
        end
        sort!(data_vector, by = x -> -x.val)
        println(data_vector[1].val)
        _random_gene_indata!(data_vector[20:end], dict_key_vector)
    end
end