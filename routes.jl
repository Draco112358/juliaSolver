using Genie, Genie.Renderer, Genie.Renderer.Html, Genie.Renderer.Json, Genie.Requests, AWS, AWSS3, SimpleWebsockets, Base.Threads, AMQPClient
include("./lib/solve.jl")

Genie.config.run_as_server = true
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "http://localhost:1212"
# This has to be this way - you should not include ".../*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

# aws_access_key_id = get(ENV, "AWS_ACCESS_KEY_ID", "")
# aws_secret_access_key = get(ENV, "AWS_SECRET_ACCESS_KEY", "")
# aws_region = get(ENV, "AWS_DEFAULT_REGION", "us-east-1")
# creds = AWSCredentials(aws_access_key_id, aws_secret_access_key)
# AWS = global_aws_config(; region=aws_region, creds=creds)

#server = WebsocketServer()

#Threads.@spawn serve(server; verbose=false)

const stopComputation = []
const commentsEnabled = []

# route("/") do
#   serve_static_file("welcome.html")
# end

# listen(server, :client) do client
#   listen(client, :message) do message
#     println(message)
#     if message == "Stop computation"
#       #error("stop computation")
#       push!(stopComputation, 1)
#     end
#   end
#   route("/solving", method="POST") do
#     # if (haskey(jsonpayload(), "mesherOutput"))
#     #   mesherOutput = jsonpayload()["mesherOutput"]
#     # else
#     if jsonpayload()["storage"] == "online"
#       p = S3Path("s3://models-bucket-49718971291/" * jsonpayload()["mesherFileId"])
#       mesherOutput = JSON.parse(read(p, String))
#     else
#       mesherOutput = JSON.parsefile(jsonpayload()["mesherFileId"])
#     end
    
    
#     # end
#     solverOutput = JSON.json(doSolving(mesherOutput, jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]; webSocketClient=client))
#     return solverOutput
#   end
# end


# route("/test_solving", method="POST") do
#   comments = length(commentsEnabled) > 0
#   if (!comments) push!(commentsEnabled, 1) end
#   return JSON.json(doSolving(jsonpayload()["mesherOutput"], jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]; commentsEnabled=comments))
# end


# function force_compile()
#   sleep(6)
#   println("------ Precompiling routes...wait for solver to be ready ---------")
#   for (name, r) in Router.named_routes()
#     data = open(JSON.parse, "first_run_data.json")
#     Genie.Requests.HTTP.request(r.method, "http://localhost:8000" * tolink(name), [("Content-Type", "application/json")], JSON.json(data))
#   end
#   println("SOLVER READY")
# end

function force_compile2()
  println("------ Precompiling routes...wait for solver to be ready ---------")
  data = open(JSON.parse, "first_run_data.json")
  doSolving(data["mesherOutput"], data["solverInput"], data["solverAlgoParams"], "init"; commentsEnabled=false)
  println("SOLVER READY")
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

#Threads.@spawn force_compile()

function publish_data(result::Dict, queue::String, chan)
  data = convert(Vector{UInt8}, codeunits(JSON.json(result)))
  message = Message(data, content_type="application/json", delivery_mode=PERSISTENT)
  basic_publish(chan, message; exchange="", routing_key=queue)
end

export publish_data

const VIRTUALHOST = "/"
const HOST = "127.0.0.1"
const stop_condition = Ref{Float64}(0.0)

function receive()
  # 1. Create a connection to the localhost or 127.0.0.1 of virtualhost '/'
  connection(; virtualhost=VIRTUALHOST, host=HOST) do conn
      # 2. Create a channel to send messages
      AMQPClient.channel(conn, AMQPClient.UNUSED_CHANNEL, true) do chan
          force_compile2()
          # EXCG_DIRECT = "MyDirectExcg"
          # @assert exchange_declare(chan1, EXCG_DIRECT, EXCHANGE_TYPE_DIRECT)
          println(" [*] Waiting for messages. To exit press CTRL+C")
          # 3. Declare a queue
          management_queue = "management_solver"
          #queue_bind(chan, "mesher_results", EXCG_DIRECT, "mesher_results")

          # 4. Setup function to receive message
          on_receive_management = (msg) -> begin
              basic_ack(chan, msg.delivery_tag)
              data = JSON.parse(String(msg.data))
              #data = String(msg.data)
              println(data["message"])
              if (data["message"] == "solving")
                println("qui")
                for (key, value) in data["body"]
                  println(key)
                end
                mesherOutput = JSON.parsefile(data["body"]["mesherFileId"])
                Threads.@spawn doSolving(mesherOutput, data["body"]["solverInput"], data["body"]["solverAlgoParams"], data["body"]["id"]; chan)
              end
              if data["message"] == "stop"
                stop_condition[] = 1.0
              end
          end

          # 5. Configure Quality of Service
          basic_qos(chan, 0, 1, false)
          success_management, consumer_tag = basic_consume(chan, management_queue, on_receive_management)

          @assert success_management == true

          while stop_condition[] != 1.0
              sleep(1)
          end
          # 5. Close the connection
      end
  end
end


# Don't exit on Ctrl-C
Base.exit_on_sigint(false)
try
  receive()
catch ex
  if ex isa InterruptException
      println("Interrupted")
  else
      println("Exception: $ex")
  end
end