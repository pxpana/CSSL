FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
