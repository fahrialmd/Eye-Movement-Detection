xhost +local:docker
docker run --gpus all -it --rm \
  --device=/dev/video0:/dev/video0 \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --privileged \
  --name skripsicontainer \
  skripsi:latest
