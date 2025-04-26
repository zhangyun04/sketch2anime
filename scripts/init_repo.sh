#!/bin/bash

# Initialize Git repository for Sketch2Anime project
# This script sets up the repository and prepares it for pushing to GitHub

set -e  # Exit on error

# Configuration
DEFAULT_REPO_NAME="sketch2anime"
DEFAULT_REMOTE_URL="https://github.com/yourusername/sketch2anime.git"

# Parse arguments
REPO_NAME=${1:-$DEFAULT_REPO_NAME}
REMOTE_URL=${2:-$DEFAULT_REMOTE_URL}

# Root directory of the project
ROOT_DIR=$(dirname $(dirname $(readlink -f "$0")))
cd $ROOT_DIR

echo "Initializing Git repository for $REPO_NAME at $ROOT_DIR"

# Initialize Git repository if it doesn't exist
if [ ! -d .git ]; then
    git init
    echo "Git repository initialized."
else
    echo "Git repository already exists."
fi

# Create .gitignore file
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
ENV/

# Models and data
models/*
!models/.gitkeep
data/*
!data/.gitkeep
results/

# Logs
logs/
*.log

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Editor configurations
.vscode/
.idea/
*.swp
*.swo
EOL

# Create necessary directories and placeholder files
mkdir -p models data logs results
touch models/.gitkeep data/.gitkeep logs/.gitkeep results/.gitkeep

# Set Git remote
if git remote | grep -q "^origin$"; then
    echo "Remote 'origin' already exists. Updating URL to $REMOTE_URL"
    git remote set-url origin $REMOTE_URL
else
    echo "Setting remote 'origin' to $REMOTE_URL"
    git remote add origin $REMOTE_URL
fi

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py
chmod +x data_processing/*.py
chmod +x train/*.py
chmod +x inference/*.py

# Stage files
git add .

echo "Repository initialized successfully!"
echo "Next steps:"
echo "1. Commit your changes:   git commit -m 'Initial commit'"
echo "2. Push to remote:        git push -u origin main" 