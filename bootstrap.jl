(pwd() != @__DIR__) && cd(@__DIR__) # allow starting app from bin/ dir

using JuliaSolver
const UserApp = JuliaSolver
JuliaSolver.main()
