# Enter virtual environment
venv/Scripts/activate

# Install package
pip install -r requirements.txt

# Add a new package to requirement.txt
pip freeze > requirements.txt
