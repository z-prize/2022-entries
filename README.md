# 2022-entries

## Summary

ZPrize 2022 - All qualified entries

> NOTE: this fork provides a Docker image to easy build, run, and profile MSM GPU implementations.

## Docker

The [Docker image](./Dockerfile) allows for building and running `MSM GPU` implementations.

```bash
# start the container
docker run -d \
   -it \
   --name zprize22-msm-gpu \
   --gpus all \
   --mount type=bind,source=$(pwd)/open-division/prize1-msm/prize1a-msm-gpu/combined-top-solutions,target=/home \
   --privileged \
   zprize22-msm-gpu:latest

# run the build
docker exec -it zprize22-msm-gpu bash
cd /home
cargo build --release
```