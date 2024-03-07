using Genie, Genie.Renderer, Genie.Renderer.Html, Genie.Renderer.Json, Genie.Requests, AWS, AWSS3
using ProfileView

Genie.config.run_as_server = true
Genie.config.cors_headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
# This has to be this way - you should not include ".../*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] ="GET,POST,PUT,DELETE,OPTIONS" 
Genie.config.cors_allowed_origins = ["*"]

AWS = global_aws_config(; region="us-east-1")



route("/") do
  serve_static_file("welcome.html")
end

route("/electron") do
  return "Prova api Genie - Electron!"
end

route("/solving" ,method="POST") do 
  p = S3Path("s3://models-bucket-49718971291/"*jsonpayload()["mesherFileId"])
  mesherOutput = JSON.parse(read(p, String))
  return JSON.json(doSolving(mesherOutput, jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]))
end

route("/test_solving" ,method="POST") do 
  return JSON.json(doSolving(jsonpayload()["mesherOutput"], jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"]))
end