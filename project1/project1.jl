using Graphs
using Printf
using CSV
using DataFrames
using SpecialFunctions
using LinearAlgebra
using Graphs  # for DiGraph and add_edge!
using TikzGraphs   # for TikZ plot output
using TikzPictures # to save TikZ as PDF
using GraphPlot
using Compose, Cairo

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
    end


function bayesian_score(vars, G, D)
    n = length(vars)
    M = statistics(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

struct Variable
    name::Symbol
    r::Int # number of possible values
end
    

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end


struct K2Search
    ordering::Vector{Int} # variable ordering
end

function fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y′ = bayesian_score(vars, G, D)
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
    end
    return G
end
    


function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    
    # read in data 
    data = CSV.File(infile) |> DataFrame
    # show(data)
    D = Matrix(Matrix(values(data))')

    # construct vars 
    variable_names = names(data)
    vars = []
    for v in variable_names
        var = Variable(Symbol(v), maximum(data[!, v]))
        push!(vars, var)
    end

    # do k2 search 
    ordering=1:length(vars)
    method = K2Search(ordering)
    G = fit(method, vars, D)

    # write out graph
    idx2names = Dict((i, vars[i].name) for i in 1:length(vars))
    write_gph(G, idx2names, outfile)
    # p = plot(G, variable_names) # create TikZ plot with labels
    # save(PDF(outfile*".pdf"), p) # save TikZ as PDF
    p = gplot(G; nodelabel=variable_names, layout=circular_layout)
    draw(PNG(outfile*".png"),p)

    # Print out final Bayesian score
    println("Final Bayesian Score: ", bayesian_score(vars,G,D))
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

@time begin
    compute(inputfilename, outputfilename)
end