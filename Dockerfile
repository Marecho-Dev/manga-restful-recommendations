# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Add this line to your existing Dockerfile to install Uvicorn
RUN pip install uvicorn


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install Node.js and NPM
RUN apt-get update && apt-get install -y curl && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

# Install Prisma CLI
RUN npm install -g prisma

#Generate the prisma client
RUN npx prisma generate

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World




# Replace the existing CMD line with this one
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "80", "optimized_manga_rec:app"]

