from datetime import datetime
from flask_login import UserMixin
from app.extensions import db
from app import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    profile_pic = db.Column(db.String(100), nullable=True, default=None) 
    created_at = db.Column(db.DateTime, default=datetime.now)  # ✅ local time
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)  # ✅ local time
    survey_completed = db.Column(db.Boolean, default=False, nullable=False)
    
    def get_id(self):
        return str(self.user_id)


class BMIRecord(db.Model):
    __tablename__ = 'bmi_records'

    bmi_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    bmi_value = db.Column(db.Float)
    bmi_category = db.Column(db.String(50))
    recorded_at = db.Column(db.DateTime, default=datetime.now)  # ✅ local time


class PerformanceAnalysis(db.Model):
    __tablename__ = 'performance_analysis'

    analysis_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    reps_completed = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Float, nullable=False)
    target_calories = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=False)
    session_date = db.Column(db.DateTime, default=datetime.now)  # ✅ local time

    user = db.relationship('User', backref='performance_analyses')


class WorkoutSchedule(db.Model):
    __tablename__ = 'workout_schedule'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    day = db.Column(db.String(10), nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    title = db.Column(db.String(100), nullable=False)

    user = db.relationship('User', backref='workout_schedules')
