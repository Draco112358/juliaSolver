using Genie, Genie.Renderer, Genie.Renderer.Html, Genie.Renderer.Json, Genie.Requests, AWS, AWSS3, SimpleWebsockets, Base.Threads
include("./lib/solve.jl")

Genie.config.run_as_server = true
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "http://localhost:1212"
# This has to be this way - you should not include ".../*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

aws_access_key_id = get(ENV, "AWS_ACCESS_KEY_ID", "")
aws_secret_access_key = get(ENV, "AWS_SECRET_ACCESS_KEY", "")
aws_region = get(ENV, "AWS_DEFAULT_REGION", "us-east-1")
creds = AWSCredentials(aws_access_key_id, aws_secret_access_key)
AWS = global_aws_config(; region=aws_region, creds=creds)

server = WebsocketServer()

@async serve(server; verbose=false)

const stopComputation = []

# route("/") do
#   serve_static_file("welcome.html")
# end

listen(server, :client) do client
  listen(client, :message) do message
    println(message)
    if message == "Stop computation"
      #error("stop computation")
      push!(stopComputation, 1)
    end
  end
  route("/solving", method="POST") do
    # if (haskey(jsonpayload(), "mesherOutput"))
    #   mesherOutput = jsonpayload()["mesherOutput"]
    # else
    p = S3Path("s3://models-bucket-49718971291/" * jsonpayload()["mesherFileId"])
    mesherOutput = JSON.parse(read(p, String))
    # end
    return JSON.json(doSolving(mesherOutput, jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]; webSocketClient=client))
  end
end


route("/test_solving", method="POST") do
  return JSON.json(doSolving(jsonpayload()["mesherOutput"], jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]))
end


function force_compile()
  sleep(6)
  println("------ Precompiling routes...wait for solver to be ready ---------")
  for (name, r) in Router.named_routes()
    data = open(JSON.parse, "first_run_data.json")
    Genie.Requests.HTTP.request(r.method, "http://localhost:8000" * tolink(name), [("Content-Type", "application/json")], JSON.json(data))
  end
  println("------------- SOLVER READY ---------------")
end

# function force_compile()
#   sleep(6)
#   println("------ Precompiling routes...wait for solver to be ready ---------")
#   client = WebsocketClient()
#   ended = Condition()
#   listen(client, :connect) do ws
#     data = open(JSON.parse, "first_run_data.json")
#     data["comments"] = true
#     Genie.Requests.HTTP.request("POST", "http://localhost:8000/solving", [("Content-Type", "application/json")], JSON.json(data))
#     notify(ended)
#   end
#   listen(client, :connectError) do err
#     notify(ended, err, error=true)
#   end

#   open(client, "ws://localhost:8080")
#   wait(ended)
#   println("------------- SOLVER READY ---------------")
# end

@async force_compile()