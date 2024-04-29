# Place here configuration options that will be set for all environments
ENV["GENIE_ENV"] = "prod"
using DotEnv
cfg = DotEnv.config()
ENV["AWS_ACCESS_KEY_ID"]=cfg["AWS_ACCESS_KEY_ID"]
ENV["AWS_SECRET_ACCESS_KEY"]=cfg["AWS_SECRET_ACCESS_KEY"]
ENV["AWS_DEFAULT_REGION"]=cfg["AWS_DEFAULT_REGION"]