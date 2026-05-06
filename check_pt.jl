using PrettyTables
println("Valid formats in PrettyTables:")
for n in names(PrettyTables; all=true)
    s = string(n)
    if occursin("tf_", s)
        println(s)
    end
end
