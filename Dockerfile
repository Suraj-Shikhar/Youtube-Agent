# Use an official Ubuntu runtime as a base image
FROM ubuntu

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Create a directory to store downloaded videos
RUN mkdir -p /app/src
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy the contents of the local src folder into the container /app/src
COPY ./src /app/src

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/

# Make the script executable
RUN chmod +x entrypoint.sh

# Set the entry point to the shell script
ENTRYPOINT ["/app/entrypoint.sh"]
