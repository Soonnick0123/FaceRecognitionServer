# Enter virtual environment
cd RecognitionServer

venv/Scripts/activate

# Install package
pip install -r requirements.txt

# Run Server
python manage.py runserver

# Add a new package to requirement.txt
pip freeze > requirements.txt
