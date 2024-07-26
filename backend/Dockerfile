FROM python3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]