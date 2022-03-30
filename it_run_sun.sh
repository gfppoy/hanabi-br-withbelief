#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_rl_games_GPU_all_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

NV_GPU="0,1,4" nvidia-docker run -it --memory="150g" \
    -e PYTHONPATH=/sad_lib \
    --cpus 20 \
    --name $name \
    --cap-add=SYS_PTRACE \
    --net host \
    --user 0 \
    -v `pwd`:/sad \
    -v `pwd`/pyhanabi:/pyhanabi \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -it sad-legacy \
    ${@:1}
