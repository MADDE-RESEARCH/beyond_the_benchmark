######################################################################
######## Official PyTorch 2.5.1, CUDA 12.4 image as the base #########
######################################################################

# Use NVIDIA PyTorch image as the base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp

    # Update package lists and install Git
RUN apt-get update && apt-get install -y git

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/*