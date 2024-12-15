# Hello Pydantic AI

This project demonstrates the usage of Pydantic with AI integration.

## Setup Instructions

1. Create and activate a virtual environment:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Set up environment variables:
- Copy the `.env.example` file to `.env` (if not already done)
- Update the values in `.env` with your configuration

## Running the Application

1. Run the main application:
```bash
python3 main.py
```

2. Run the customer module:
```bash
python3 customer.py
```

## Project Structure
- `main.py`: Main application entry point
- `customer.py`: Customer-related functionality
- `.env`: Environment configuration file
- `requirements.txt`: Project dependencies

## Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`
