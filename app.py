from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# MySQL Configuration
app.config['MYSQL_HOST'] = os.getenv('DB_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('DB_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('DB_PASSWORD', 'password')
app.config['MYSQL_DB'] = os.getenv('DB_NAME', 'customer_segmentation')
app.config['MYSQL_PORT'] = int(os.getenv('DB_PORT', 3306))
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = mysql.connect
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        if user:
            return User(id=user['id'], username=user['username'], email=user['email'])
        return None
    except Exception as e:
        print(f"Error loading user: {str(e)}")
        return None

def create_database():
    try:
        connection = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            port=app.config['MYSQL_PORT']
        )
        
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{app.config['MYSQL_DB']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            connection.commit()
        
        connection.close()
        print(f"Database {app.config['MYSQL_DB']} created/verified")
        return True
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        return False

def create_tables():
    try:
        connection = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
        )
        
        with connection.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS `users` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `username` VARCHAR(50) NOT NULL UNIQUE,
                `password` VARCHAR(255) NOT NULL,
                `email` VARCHAR(100) NOT NULL UNIQUE,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            cursor.execute("SHOW TABLES LIKE 'users'")
            if not cursor.fetchone():
                raise Exception("Users table not created")
                
        connection.commit()
        connection.close()
        print("Database tables verified")
        return True
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        return False

def initialize_database():
    print("Initializing database...")
    if not create_database():
        return False
    if not create_tables():
        return False
    return True

# Load model and scaler
try:
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("ML models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)

# Data storage for visualization
data_store = []

def validate_registration(username, email, password):
    """Validate registration data"""
    errors = []
    
    # Username validation
    if not username or not username[0].isupper():
        errors.append('Username must start with a capital letter')
    elif not re.match(r'^[A-Z][a-zA-Z0-9]*$', username):
        errors.append('Username can only contain letters and numbers')
    
    # Email validation
    if not email or not email.endswith('@gmail.com'):
        errors.append('Please use a valid Gmail address')
    
    # Password validation
    if len(password) < 8:
        errors.append('Password must be at least 8 characters')
    elif not re.search(r'[!@#$%^&*]', password):
        errors.append('Password must contain at least one special character (!@#$%^&*)')
    
    return errors

# Routes
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template("index.html", prediction=None, images=None)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Please fill in all fields', 'danger')
            return render_template('login.html')
        
        try:
            conn = mysql.connect
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            cur.close()
            
            if user and check_password_hash(user['password'], password):
                user_obj = User(id=user['id'], username=user['username'], email=user['email'])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'danger')
        except Exception as e:
            flash('Login failed. Please try again.', 'danger')
            print(f"Login error: {str(e)}")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        # Validate inputs
        errors = validate_registration(username, email, password)
        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')
        
        try:
            conn = mysql.connect
            cur = conn.cursor()
            
            cur.execute("SELECT username, email FROM users WHERE username = %s OR email = %s", 
                       (username, email))
            existing = cur.fetchone()
            
            if existing:
                if existing['username'] == username:
                    flash('Username already exists', 'danger')
                else:
                    flash('Email already exists', 'danger')
            else:
                hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
                cur.execute(
                    "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                    (username, hashed_pw, email)
                )
                conn.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
                
        except pymysql.err.IntegrityError as e:
            flash('Username or email already exists', 'danger')
            print(f"Integrity error: {str(e)}")
        except pymysql.err.DataError as e:
            flash('Input data too long. Please use shorter values.', 'danger')
            print(f"Data error: {str(e)}")
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'danger')
            print(f"Registration error: {str(e)}")
        finally:
            if 'cur' in locals():
                cur.close()
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        age = int(request.form.get("age", 0))
        income = int(request.form.get("income", 0))
        score = int(request.form.get("score", 0))
        
        if not all([age > 0, income > 0, 0 < score <= 100]):
            raise ValueError("Invalid input values")
        
        data = np.array([[age, income, score]])
        data_scaled = scaler.transform(data)
        cluster = kmeans.predict(data_scaled)[0]
        
        data_store.append({"Age": age, "Income": income, "Score": score, "Cluster": cluster})
        
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Generate visualizations
        images = generate_visualizations()
        
        return render_template("index.html", 
                             prediction=f"Customer belongs to Cluster {cluster}", 
                             images=images)
    except Exception as e:
        flash(f'Error processing your request: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

def generate_visualizations():
    if not data_store:
        return None
    
    df = pd.DataFrame(data_store)
    images = {}
    
    try:
        # 1. Income vs. Spending Score
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="Income", y="Score", hue="Cluster", 
                       data=df, palette="viridis", s=100)
        plt.title("Income vs. Spending Score")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        img_path1 = os.path.join('static', 'income_vs_score.png')
        plt.savefig(img_path1, bbox_inches='tight', dpi=100)
        plt.close()
        images["income_vs_score"] = 'income_vs_score.png'
        
        # 2. Age vs. Spending Score
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="Age", y="Score", hue="Cluster",
                       data=df, palette="coolwarm", s=100)
        plt.title("Age vs. Spending Score")
        plt.xlabel("Age")
        plt.ylabel("Spending Score (1-100)")
        img_path2 = os.path.join('static', 'age_vs_score.png')
        plt.savefig(img_path2, bbox_inches='tight', dpi=100)
        plt.close()
        images["age_vs_score"] = 'age_vs_score.png'
        
        # 3. Age vs. Income
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="Age", y="Income", hue="Cluster",
                       data=df, palette="Set1", s=100)
        plt.title("Age vs. Income")
        plt.xlabel("Age")
        plt.ylabel("Annual Income (k$)")
        img_path3 = os.path.join('static', 'age_vs_income.png')
        plt.savefig(img_path3, bbox_inches='tight', dpi=100)
        plt.close()
        images["age_vs_income"] = 'age_vs_income.png'
        
        # 4. Cluster Distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x="Cluster", data=df, palette="pastel")
        plt.title("Cluster Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Customers")
        img_path4 = os.path.join('static', 'cluster_distribution.png')
        plt.savefig(img_path4, bbox_inches='tight', dpi=100)
        plt.close()
        images["cluster_distribution"] = 'cluster_distribution.png'
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return None
    
    return images

if __name__ == '__main__':
    with app.app_context():
        if not initialize_database():
            print("Database initialization failed! Please check MySQL server and credentials.")
            print("Make sure MySQL is running and the user has proper privileges.")
            print("You may need to manually create the database first.")
            exit(1)
    
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True)