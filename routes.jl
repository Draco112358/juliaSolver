using Genie, Genie.Renderer, Genie.Renderer.Html, Genie.Renderer.Json, Genie.Requests, AWS, AWSS3, SimpleWebsockets
include("./lib/solve.jl")

Genie.config.run_as_server = true
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "http://localhost:1212"
# This has to be this way - you should not include ".../*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] ="GET,POST,PUT,DELETE,OPTIONS" 
Genie.config.cors_allowed_origins = ["*"]

aws_access_key_id = get(ENV,"AWS_ACCESS_KEY_ID","")
aws_secret_access_key = get(ENV, "AWS_SECRET_ACCESS_KEY","")
aws_region = get(ENV,"AWS_DEFAULT_REGION","us-east-1")
creds = AWSCredentials(aws_access_key_id, aws_secret_access_key)
AWS = global_aws_config(; region=aws_region, creds=creds)

server = WebsocketServer()

@async serve(server; verbose = false)


route("/") do
  serve_static_file("welcome.html")
end

route("/electron") do
  return "Prova api Genie - Electron!"
end

listen(server, :client) do client 
  route("/solving" ,method="POST") do 
    p = S3Path("s3://models-bucket-49718971291/"*jsonpayload()["mesherFileId"])
    mesherOutput = JSON.parse(read(p, String))
    return JSON.json(doSolving(mesherOutput, jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"], client))
  end
end

# route("/test_solving" ,method="POST") do 
#   println("-------------------- SOLVING... -----------------------")
#   solving_result = doSolving(jsonpayload()["mesherOutput"], jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"])
#   println("-------------------- SOLVING FINISHED! -----------------------")
#   return JSON.json(solving_result)
# end