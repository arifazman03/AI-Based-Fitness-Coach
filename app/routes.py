from datetime import datetime, timedelta
from flask import Blueprint, jsonify, render_template, redirect, url_for, flash, request, Response, session, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app.forms import RegisterForm, LoginForm, BMIForm, EditProfileForm
from app.models import User, BMIRecord, PerformanceAnalysis, WorkoutSchedule
from dotenv import load_dotenv
from app import db
import cv2
import mediapipe as mp
import numpy as np
import time
import io
import base64
import matplotlib.pyplot as plt
import secrets
import os
from PIL import Image
import random
import requests



# Initialize Blueprint
main = Blueprint('main', __name__)

# Global variables for webcam and exercise counter
# It's generally better to pass these around or use a more sophisticated state management,
# but for a simple Flask app, global variables can work with caution.
cap = None
counter = None

# Time Slot Mapping (will be used to fill the 'time' column in schedule)
TIME_SLOTS = {
    'A': '7:00 AM',  
    'B': '1:00 PM',  
    'C': '5:00 PM',  
    'D': '9:00 PM',  
    'E': 'Flexible (F)' 
}

# Exercise Key (Dumbbell Exercises)
EXERCISE_KEY = {
    'PUSH': 'Floor Press, Shoulder Press, Tricep Ext',
    'PULL': 'Row, Bicep Curl, Reverse Fly',
    'LEGS': 'Squat, RDL, Lunge',
    'UPPER': 'Shoulder Press, DB Row, Bicep Curl',
    'FULL': 'Squat, Push Press, Row',
    'ARMS': 'Bicep Curl, Hammer Curl, Tricep Ext',
    'SHOULDERS': 'Shoulder Press, Lateral Raise, Front Raise',
    'REST': 'Rest'
}

# Weekly Schedule Templates
SCHEDULE_TEMPLATES = {
    'A': ['PUSH', 'PULL', 'LEGS', 'REST', 'PUSH', 'PULL', 'REST'],  
    'B': ['FULL', 'REST', 'FULL', 'REST', 'FULL', 'REST', 'REST'],    
    'C': ['UPPER', 'LEGS', 'REST', 'UPPER', 'LEGS', 'REST', 'REST'],  
    'D': ['LEGS', 'ARMS', 'REST', 'SHOULDERS', 'REST', 'LEGS', 'REST'] 
}

# default Start & End Times for Generated Workouts
DEFAULT_WORKOUT_DURATION = 60 # in minutes
DEFAULT_START_TIME = '10:00' # for answer 'Anytime or Flexible' in Q4

def generate_and_save_schedule(q4_time_code, q5_split_code, user_id):
    """Generates a 7-day schedule based on survey answers and saves it to the DB."""
    
    # get the split template and time slot preference
    schedule_template = SCHEDULE_TEMPLATES.get(q5_split_code, SCHEDULE_TEMPLATES['B'])
    time_preference = TIME_SLOTS.get(q4_time_code, TIME_SLOTS['E'])
    
    # define  consistent start time for non-flexible/non-rest days
    try:
        # Attempt to parse time from TIME_SLOTS if possible
        if 'AM' in time_preference or 'PM' in time_preference:
            time_part = time_preference.split(' ')[0]
            start_dt = datetime.strptime(time_part, '%H:%M')
        else: # Flexible or fallback
            start_dt = datetime.strptime(DEFAULT_START_TIME, '%H:%M')
    except ValueError:
        start_dt = datetime.strptime(DEFAULT_START_TIME, '%H:%M')
    
    # convert to time object
    start_time = start_dt.time()
    
    new_items = []
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day_name, workout_type_key in zip(day_names, schedule_template):
        if workout_type_key == 'REST':
            # Create a "Rest" entry in the schedule
            pass
        else:
            # generate the workout title
            exercises = EXERCISE_KEY.get(workout_type_key, 'General Workout')
            
            # create workout title with time preference (for flexible users)
            if 'Flexible' in time_preference:
                title = f"({time_preference}) {workout_type_key} - {exercises}"
            else:
                # calculate end time: Start + 60 mins
                end_dt = start_dt + timedelta(minutes=DEFAULT_WORKOUT_DURATION)
                
                # update start/end time for non-flexible days
                start_time_obj = start_dt.time()
                end_time_obj = end_dt.time()
                
                title = f"{workout_type_key} - {exercises}"
                
            new_item = WorkoutSchedule(
                user_id=user_id,
                day=day_name,
                # store the actual time only if it's not flexible.
                start_time=start_time_obj if 'Flexible' not in time_preference else datetime.time(10, 0), # use default time for flexible days
                end_time=end_time_obj if 'Flexible' not in time_preference else time(11, 0), # use default time for flexible days
                title=title
            )
            new_items.append(new_item)

    # delete existing items before saving the new items
    WorkoutSchedule.query.filter_by(user_id=user_id).delete()
    db.session.add_all(new_items)
    db.session.commit()
    
    return True

# ---  Survey Route  ---
@main.route('/survey', methods=['GET', 'POST'])
@login_required
def survey():
    if request.method == 'POST':
        # retrieve Q4 and Q5
        q4_workout_time = request.form.get('q4_workout_time')
        q5_workout_split = request.form.get('q5_workout_split')
        
        # for validation
        if not q4_workout_time or not q5_workout_split:
            flash('Please answer all questions before submitting.', 'error')
            return redirect(url_for('main.survey'))

        # generate and save the 7 day plan to the database
        generate_and_save_schedule(q4_workout_time, q5_workout_split, current_user.user_id)
        
        return redirect(url_for('main.workout_schedule'))
    
    return render_template('survey.html')


# ----------------------------------------------------------------------
## Helper Class for Exercise Counting (replace original launch_camera logic)
# ----------------------------------------------------------------------
class ExerciseCounter:
    def __init__(self, exercise_type):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.exercise_type = exercise_type
        self.count = 0
        self.stage = None # "up" or "down"
        self.start_time = time.time() # For future calorie calculation based on duration if needed

    def get_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Make image not writeable for performance
        results = self.pose.process(image)
        image.flags.writeable = True # Make image writeable for drawing
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            if self.exercise_type == "bicep_curl":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # You might also want to track the right arm
                # R_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                # R_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                # R_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                angle = self.get_angle(shoulder, elbow, wrist)
                # Visual feedback for angle
                cv2.putText(image, str(int(angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Rep counter logic
                if angle > 160:
                    self.stage = "down"
                if angle < 50 and self.stage == "down":
                    self.stage = "up"
                    self.count += 1

            elif self.exercise_type == "shoulder_press":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = self.get_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(int(angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160: # Arm extended (down position)
                    self.stage = "down"
                if angle < 90 and self.stage == "down": # Arm bent (up position)
                    self.stage = "up"
                    self.count += 1

            elif self.exercise_type == "squat":
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angle = self.get_angle(hip, knee, ankle)
                cv2.putText(image, str(int(angle)), tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160: # Standing up
                    self.stage = "up"
                if angle < 90 and self.stage == "up": # Squatting down
                    self.stage = "down"
                    self.count += 1

            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                           self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                          )
            
        except Exception as e:
            # print(f"Error processing landmarks: {e}") # For debugging
            pass # Continue if landmarks aren't detected

        # Display rep count
        cv2.putText(image, f'Reps: {self.count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        return image, self.count
    
# ----------------------------------------------------------------------
## Helper Function for Profile Picture Upload
# ----------------------------------------------------------------------
def save_picture(form_picture):
    """
    Saves the uploaded profile picture, resizing it, and returns the unique filename.
    """
    # 1. generate a random filename
    random_hex = secrets.token_hex(8)
    # Get the file extension (e.g., .png, .jpg)
    _, f_ext = os.path.splitext(form_picture.filename) 
    picture_fn = random_hex + f_ext
    
    # 2. get UPLOAD_FOLDER path from the running app config (set in __init__.py)
    # NOTE: You MUST have current_app.config['UPLOAD_FOLDER'] set up to point to the 'static/profile_pics' directory.
    upload_path = current_app.config['UPLOAD_FOLDER'] 
    
    # 3. define full path to save the picture
    picture_path = os.path.join(upload_path, picture_fn)

    # 4. resize image 
    output_size = (150, 150) 
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    
    # 5. save image
    i.save(picture_path)

    return picture_fn
# ----------------------------------------------------------------------

# Load the variables from .env file
load_dotenv()
print(f"KEY LOADED: {os.getenv('GEMINI_API_KEY')[:10]}...") # Prints just the start of the key for safety

# ----------------------------------------------------------------------
## Flask Routes
# ----------------------------------------------------------------------

@main.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    else:
        return redirect(url_for('main.login'))

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if username already exists
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists! Please choose another one.', 'danger')
            return render_template('register.html', form=form)

        # Check if passwords match
        if form.password.data != form.confirm_password.data:
            flash('Passwords do not match. Please try again.', 'danger')
            return render_template('register.html', form=form)

        # Create new user
        hashed_pw = generate_password_hash(form.password.data)
        new_user = User(
            name=form.name.data,
            gender=form.gender.data,
            age=form.age.data,
            username=form.username.data,
            password=hashed_pw,
            profile_pic='no_pfp.jpg'  # Default profile picture
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('main.login'))

    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('main.dashboard')) 
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('main.login'))

@main.route('/profile') 
@login_required 
def profile():
    return render_template('profile.html', user=current_user)

@main.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(original_username=current_user.username)

    if form.validate_on_submit():
        # 1. handle pfp update
        if form.picture.data:
            try:
                # delete old picture if not default
                if current_user.profile_pic and current_user.profile_pic != 'no_pfp.jpg':
                    old_pic_path = os.path.join(current_app.config['UPLOAD_FOLDER'], current_user.profile_pic)
                    if os.path.exists(old_pic_path):
                        os.remove(old_pic_path)

                # save new picture
                picture_file = save_picture(form.picture.data)
                current_user.profile_pic = picture_file
                # flash('Your profile picture has been updated!', 'success')
            except Exception as e:
                # log the error and notify user
                current_app.logger.error(f"Error saving profile picture for user {current_user.username}: {e}")
                flash('An error occurred while uploading your profile picture.', 'danger')
                return redirect(url_for('main.edit_profile'))

        # 2. update profile details
        current_user.name = form.name.data
        current_user.gender = form.gender.data
        current_user.age = form.age.data
        current_user.username = form.username.data

        # 3. update password if provided
        if form.password.data:
            current_user.password = generate_password_hash(form.password.data)
        
        # 4. commit all changes
        db.session.commit()
        flash('Your profile has been updated!', 'success')
        return redirect(url_for('main.profile'))

    elif request.method == 'GET':
        form.name.data = current_user.name
        form.gender.data = current_user.gender
        form.age.data = current_user.age
        form.username.data = current_user.username
        
    return render_template('edit_profile.html', form=form, user=current_user)

@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name)

@main.route('/bmi', methods=['GET', 'POST'])
@login_required
def bmi():
    form = BMIForm()
    bmi_result = None
    bmi_category = None

    if form.validate_on_submit():
        height_m = form.height_cm.data / 100
        bmi = round(form.weight_kg.data / (height_m ** 2), 2)

        # Determine category
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        # Save to database
        new_bmi = BMIRecord(
            user_id=current_user.user_id, # Use current_user.id as defined in your User model
            height_cm=form.height_cm.data,
            weight_kg=form.weight_kg.data,
            bmi_value=bmi,
            bmi_category=category
        )
        db.session.add(new_bmi)
        db.session.commit()

        bmi_result = bmi
        bmi_category = category

    return render_template('bmi.html', form=form, bmi_result=bmi_result, bmi_category=bmi_category)

@main.route('/tutorial_initial') # Instruction page
@login_required 
def tutorial_initial():
    """
    Renders the tutorial instruction page (tutorial_initial.html).
    """
    return render_template('tutorial_initial.html')

@main.route('/tutorial_biceps')
@login_required
def tutorial_biceps():
    return render_template('tutorial_biceps.html')

@main.route('/tutorial_shoulder')
@login_required
def tutorial_shoulder():
    return render_template('tutorial_shoulder.html')

@main.route('/tutorial_squats')
@login_required
def tutorial_squats():
    return render_template('tutorial_squats.html')

@main.route('/trainingintl') 
@login_required 
def trainingintl():
    """
    Renders the training instruction page (trainingintl.html).
    This page displays information about getting started with the AI Fitness Coach.
    """
    return render_template('trainingintl.html')

# This route will now ONLY render the form on GET requests.
# The form itself posts directly to /start_training.
@main.route('/training', methods=['GET']) # Removed 'POST' method here
@login_required
def training():
    """
    Renders the page where the user can choose an exercise and input training parameters.
    This is the primary /training route.
    """
    return render_template('training.html')

# This route will now handle both GET and POST.
# POST: from the form submission on training.html
# GET: if redirected here or accessed directly
@main.route('/start_training', methods=['GET', 'POST'])
@login_required
def start_training():
    global cap, counter

    if request.method == 'POST':
        # Retrieve form data and store in session
        session['exercise'] = request.form.get('exercise')
        # Ensure conversion to float for safety, default to 0.0 if missing/invalid
        try:
            session['weight'] = float(request.form.get('weight'))
            session['target_calories'] = float(request.form.get('target_calories'))
        except (ValueError, TypeError):
            flash("Invalid input for weight or target calories. Please enter numbers.", 'danger')
            return redirect(url_for('main.training'))

        # Print for debugging
        print(f"POST request to /start_training received. Exercise: {session['exercise']}")

        # Initialize camera and counter
        # Ensure cap is released and re-initialized if a previous session was active
        if cap is not None and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            cap = None # Reset cap

        cap = cv2.VideoCapture(0) # 0 for default webcam
        if not cap.isOpened():
            flash('Could not access webcam. Please ensure it is connected and not in use.', 'danger')
            return redirect(url_for('main.training')) # Redirect back to training selection

        counter = ExerciseCounter(session['exercise'])
        return render_template('start_training.html') # This HTML will contain the video feed

    elif request.method == 'GET':
        # If a GET request is received (e.g., direct navigation or a refresh/redirect after POST),
        # ensure we have session data to proceed, otherwise redirect to the form page.
        if 'exercise' in session and 'weight' in session and 'target_calories' in session:
            # This handles cases where a user refreshes the /start_training page
            print("GET request to /start_training with existing session data.")

            # Re-initialize camera if it's not active
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    flash('Could not access webcam. Please ensure it is connected and not in use.', 'danger')
                    return redirect(url_for('main.training'))

            # Re-initialize counter if it's None 
            if counter is None:
                 counter = ExerciseCounter(session['exercise'])

            return render_template('start_training.html')
        else:
            # If no session data, it means direct GET access without prior form submission.
            print("GET request to /start_training without session data. Redirecting to choose_exercise page.")
            flash("Please choose an exercise from the training page to start a session.", 'info')
            return redirect(url_for('main.training')) # Redirect to the choose exercise page


def generate_frames():
    global cap, counter
    while True:
        if cap is None or not cap.isOpened():
            break # Exit if camera is not available or closed

        success, frame = cap.read()
        if not success:
            break

        # Process the frame with the ExerciseCounter
        processed_frame, current_reps = counter.process_frame(frame)
        
        # Encode the processed frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release resources when done generating frames
    if cap:
        cap.release()
    cv2.destroyAllWindows()


@main.route('/video_feed')
@login_required
def video_feed():
    # Return the stream generated by generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/stop_training', methods=['POST'])
@login_required
def stop_training():
    global cap, counter
    if cap:
        cap.release() # Release the webcam
        cap = None # Set to None to indicate it's closed
    cv2.destroyAllWindows() # Close any OpenCV windows

    reps = counter.count if counter else 0
    weight = session.get('weight', 0)
    exercise = session.get('exercise', 'unknown')

    # Calculate calories burned
    calorie_factors = {
        "bicep_curl": 0.35,
        "squat": 0.50,
        "shoulder_press": 0.45
    }
    # Fallback to a default factor if exercise type isn't found
    calories = round(reps * calorie_factors.get(exercise, 0.35) * (weight / 60), 2)
    
    target_calories = session.get('target_calories', 0)
    
    # Save training record to database
    training_record = PerformanceAnalysis(
        user_id=current_user.user_id,
        exercise_type=exercise,
        reps_completed=reps,
        calories_burned=calories,
        target_calories=target_calories,
        weight_kg=weight
    )
    db.session.add(training_record)
    db.session.commit()

    # Store final results in session to pass to the result page
    session['reps'] = reps
    session['calories'] = calories
    
    # Clear the counter instance
    counter = None 

    return redirect(url_for('main.training_result'))

@main.route('/training_result')
@login_required
def training_result():
    # Retrieve results from session
    reps = session.pop('reps', 0)
    calories = session.pop('calories', 0)
    target_calories = session.pop('target_calories', 0)
    exercise = session.pop('exercise', 'Exercise')
    session.pop('weight', None)

    # Generate the Matplotlib chart
    fig, ax = plt.subplots()
    values = [calories, target_calories]
    labels = ['Calories Burned', 'Target Calories']
    colors = ['green', 'orange']

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Calories (kcal)', fontsize=12)
    ax.set_title(f'{exercise.replace("_", " ").title()} Summary', fontsize=16, pad=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add values on top of the bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ax.get_ylim()[1] * 0.02),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    # Set Y-limit
    max_val = max(values)
    ax.set_ylim(0, max(1, max_val * 1.2)) # Corrected for edge case where max_val is 0

    # Save to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return render_template('training_result.html',
                            reps=reps,
                            calories=calories,
                            exercise=exercise,
                            calorie_target=target_calories,
                            image_base64=image_base64)

@main.route('/printl') # Instruction page
@login_required 
def printl():
    """
    Renders the progress print options page (printl.html).
    """
    return render_template('printl.html')

@main.route('/progress')
@login_required
def progress():
    # Get active_tab from query parameters, default to 'bmi'
    active_tab = request.args.get('active_tab', 'bmi')

    # Fetch BMI Records
    bmi_records = BMIRecord.query.filter_by(user_id=current_user.user_id).order_by(BMIRecord.recorded_at).all()
    bmi_dates = [record.recorded_at.strftime('%Y-%m-%d') for record in bmi_records]
    bmi_values = [record.bmi_value for record in bmi_records]
    bmi_categories = [record.bmi_category for record in bmi_records]

    selected_type = request.args.get('exercise_type', 'all')
    all_types = db.session.query(PerformanceAnalysis.exercise_type).filter_by(user_id=current_user.user_id).distinct()

    if selected_type == 'all':
        training_records = PerformanceAnalysis.query.filter_by(user_id=current_user.user_id).order_by(PerformanceAnalysis.session_date).all()
    else:
        training_records = PerformanceAnalysis.query.filter_by(user_id=current_user.user_id, exercise_type=selected_type).order_by(PerformanceAnalysis.session_date).all()
    
    session_dates = [record.session_date.strftime('%Y-%m-%d') for record in training_records]
    calories_burned = [record.calories_burned for record in training_records]
    reps_completed = [record.reps_completed for record in training_records]
    exercise_types_series = [record.exercise_type.replace('_', ' ').title() for record in training_records]
    
    return render_template('progress_records.html',
                            bmi_dates=bmi_dates,
                            bmi_values=bmi_values,
                            bmi_categories=bmi_categories,
                            session_dates=session_dates,
                            calories_burned=calories_burned,
                            reps_completed=reps_completed,
                            exercise_types_series=exercise_types_series,
                            exercise_types=[t[0] for t in all_types],
                            selected_type=selected_type,
                            active_tab=active_tab) # Pass active_tab to the template

@main.route('/workout_schedule')
@login_required
def workout_schedule():
    # Get all schedule items for the current user
    schedule_items = WorkoutSchedule.query.filter_by(user_id=current_user.user_id).all()

    # Organize items into a dictionary by day
    schedule_dict = { 'Monday': [], 'Tuesday': [], 'Wednesday': [],
    'Thursday': [], 'Friday': [], 'Saturday': [], 'Sunday': []}
    for item in schedule_items:
        formatted = f"{item.start_time.strftime('%H:%M')} - {item.end_time.strftime('%H:%M')}<br>{item.title}"
        schedule_dict[item.day].append(formatted)
        schedule_items = WorkoutSchedule.query.filter_by(user_id=current_user.user_id).all()
        items = WorkoutSchedule.query.filter_by(user_id=current_user.user_id).all()

    return render_template('workout_schedule.html', schedule=schedule_dict, schedule_items=schedule_items)


@main.route('/add_schedule_item', methods=['POST'])
@login_required
def add_schedule_item():
    day = request.form['day']
    start_time_str = request.form['start_time']
    end_time_str = request.form['end_time']
    title = request.form['title']

    from datetime import datetime

    # Convert string to time object
    start_time = datetime.strptime(start_time_str, '%H:%M').time()
    end_time = datetime.strptime(end_time_str, '%H:%M').time()

    new_item = WorkoutSchedule(
        user_id=current_user.user_id,
        day=day,
        start_time=start_time,
        end_time=end_time,
        title=title
    )
    db.session.add(new_item)
    db.session.commit()

    return redirect(url_for('main.workout_schedule'))  # adjust this to your page

@main.route('/delete_schedule_item', methods=['POST'])
@login_required
def delete_schedule_item():
    item_id = request.form.get('item_id')
    item = WorkoutSchedule.query.filter_by(id=item_id, user_id=current_user.user_id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return redirect(url_for('main.workout_schedule'))

@main.route('/edit_schedule_item', methods=['POST'])
@login_required
def edit_schedule_item():
    item_id = request.form.get('item_id')
    new_day = request.form.get('new_day')
    new_start_time = datetime.strptime(request.form.get('new_start_time'), '%H:%M').time()
    new_end_time = datetime.strptime(request.form.get('new_end_time'), '%H:%M').time()
    new_title = request.form.get('new_title')

    item = WorkoutSchedule.query.filter_by(id=item_id, user_id=current_user.user_id).first()
    if item:
        item.day = new_day
        item.start_time = new_start_time
        item.end_time = new_end_time
        item.title = new_title
        db.session.commit()

    return redirect(url_for('main.workout_schedule'))


@main.route('/clear_schedule', methods=['POST'])
@login_required
def clear_schedule():
    WorkoutSchedule.query.filter_by(user_id=current_user.user_id).delete()
    db.session.commit()
    return jsonify({'status': 'success'})

@main.route('/get_schedule')
@login_required
def get_schedule():
    schedule = WorkoutSchedule.query.filter_by(user_id=current_user.user_id).all()
    data = {}

    for item in schedule:
        day = item.day.lower()
        if day not in data:
            data[day] = []
        data[day].append({
            'start_time': item.start_time.strftime('%H:%M'),
            'end_time': item.end_time.strftime('%H:%M'),
            'title': item.title
        })

    return jsonify(data)

@main.route('/get_schedule_items_dropdown')
@login_required
def get_schedule_items_dropdown():
    schedule = WorkoutSchedule.query.filter_by(user_id=current_user.user_id).all()
    items = [{
        'id': item.id,
        'day': item.day,
        'start_time': item.start_time.strftime('%H:%M'),
        'end_time': item.end_time.strftime('%H:%M'),
        'title': item.title
    } for item in schedule]
    return jsonify(items)

@main.route('/check_today_schedule')
@login_required
def check_today_schedule():
    today = datetime.now().strftime('%A')  # e.g. "Monday", "Tuesday"
    has_schedule = WorkoutSchedule.query.filter_by(user_id=current_user.user_id, day=today).first() is not None
    return jsonify({'has_schedule': has_schedule})

@main.route('/chatbot')
def chatbot():
    """Renders the AI Chatbot page."""
    return render_template('chatbot.html')

@main.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_contents = data.get('contents', [])
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Use the stable 1.5 flash URL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": user_contents}

        response = requests.post(url, headers=headers, json=payload)
        
        # Check if we hit a rate limit
        if response.status_code == 429:
            return jsonify({
                "error": "Message limit reached. Please wait a minute before trying again."
            }), 429
        
        # Check for other errors
        if response.status_code != 200:
            return jsonify({"error": "The AI is currently unavailable."}), response.status_code
            
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": "Server error occurred."}), 500

@main.route('/logout-confirmation')
def logout_confirmation():
    return render_template('logout_confirmation.html')