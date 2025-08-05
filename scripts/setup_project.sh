#!/bin/bash

# Vec2Summ Project Setup Script
echo "🚀 Setting up Vec2Summ project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode
echo "Installing vec2summ package..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p checkpoints

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your OpenAI API key!"
fi

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Add your data to data/raw/"
echo "3. Run an experiment:"
echo "   python scripts/run_experiment.py --data_path data/raw/your_data.csv --evaluate --visualize"
echo ""
echo "For more information, see README.md"
