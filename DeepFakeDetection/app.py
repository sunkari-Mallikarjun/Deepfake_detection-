from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import cv2  # To process video frames
import os
import numpy as np
from werkzeug.utils import secure_filename
import picklegit

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '5f4dcc3b5aa765d61d8327deb882cf99'
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Set the path to store uploaded files
db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

with open(r'models\randomforest.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

mesonet_model = load_model(r'models\mesonet_finetuned.keras')
xceptionnet_model = load_model(r'models\xception_finetuned (1).keras')


processor_vit = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
vit_model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
label2id = {"Real": 0, "Fake": 1}
id2label = {v: k for k, v in label2id.items()}

mesonet_model2 = load_model(r'models\mesonet_finetuned.keras')
mesonet_model2 = Model(inputs=mesonet_model2.input, outputs=mesonet_model2.get_layer(index=-2).output) # MesoNet model
xcept_model2 = load_model(r'models\xception_finetuned (1).keras')
xcept_model2 = Model(inputs=xcept_model2.input, outputs=xcept_model2.get_layer(index=-2).output)  # Xception model

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.secret_key = '5f4dcc3b5aa765d61d8327deb882cf99'
# db = SQLAlchemy(app)

# Define your models here (User, Post, DeepfakeDetection, Comment, Like)
# User Table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(120), nullable=False)
    profile_pic = db.Column(db.String, nullable=True)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)



# Posts Table
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String, nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    is_deepfake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    
# Deepfake Detection Table
class DeepfakeDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    model_used = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    result = db.Column(db.Boolean, nullable=False)
    

# Comments Table
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Likes Table
class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    if 'user_id' in session:
        logged_in = True
        # Use a join query to get posts along with the corresponding user's username
        posts = db.session.query(Post, User.username).join(User, Post.user_id == User.id).all()
        posts[:]=posts[::-1]
    else:
        logged_in = False
        posts = []  # No posts to show if not logged in

    return render_template('home.html', logged_in=logged_in, posts=posts)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return  redirect(url_for('dashboard'))
        flash('Login Unsuccessful. Please check your email and password', 'danger')
    return render_template('login.html')

def allowed_file(filename):
    """Check if a file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return  redirect(url_for('dashboard'))
    return render_template('signup.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Check if the 'file' part is present in the request
#         if 'file' not in request.files:
#             flash('No file part', 'danger')
#             return redirect(request.url)

#         file = request.files['file']
        
#         # Check if a file is selected
#         if file.filename == '':
#             flash('No selected file', 'danger')
#             return redirect(request.url)

#         # Validate the file type
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             relative_file_path = os.path.join('upload', filename)  # Save relative path
#             absolute_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Full path

#             # Save the file in the 'upload' directory
#             file.save(absolute_file_path)

#             # Check if the file is an image or video and classify it
#             file_extension = filename.rsplit('.', 1)[1].lower()
#             if file_extension in ['png', 'jpg', 'jpeg']:
#                 # Image classification logic
#                 final_result, img_confi = image_classification_dynamic_weight(absolute_file_path)
#             elif file_extension == 'mp4':
#                 # Video classification logic
#                 final_result, vid_confi = video_classification(absolute_file_path)
#             else:
#                 flash('Unsupported file type', 'danger')
#                 os.remove(absolute_file_path)
#                 return redirect(url_for('upload'))

#             # If classified as "Real", save the post in the database
#             if final_result == "Real":
#                 try:
#                     new_post = Post(
#                         user_id=session.get('user_id'),
#                         file_path=relative_file_path,  # Save the relative path
#                         file_type=file_extension,
#                         is_deepfake=False,  # It's classified as real
#                         confidence=1.0  # Assuming 100% confidence
#                     )
#                     db.session.add(new_post)
#                     db.session.commit()
#                     flash('File uploaded successfully!', 'success')
#                 except Exception as e:
#                     # Handle database errors (e.g., rollback)
#                     db.session.rollback()
#                     flash('Error saving to the database: {}'.format(str(e)), 'danger')
#                     os.remove(absolute_file_path)
#                     return redirect(url_for('upload'))

#             # If classified as "Fake", delete the file
#             else:
#                 os.remove(absolute_file_path)
#                 flash('File is classified as fake and cannot be uploaded.', 'danger')
#                 return redirect(url_for('upload'))

#             return redirect(url_for('result'))

#     return render_template('upload.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Check if the file is an image or video
            if filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']:
                # Image classification
                # final_result, img_confi = image_classification_dynamic_weight(file_path)
                features = predict_single_image(file_path)
                nfeatures = features.reshape(1, -1)

                # Use the trained classifier (e.g., Random Forest) to predict
                prediction = clf.predict(nfeatures)
                # Final result based on classifier output
                final_result = 'Real' if prediction[0] == 0 else 'Fake'

            elif filename.rsplit('.', 1)[1].lower() == 'mp4':
                # Video classification
                final_result = video_classification(file_path)

            # If the result is "Real", allow the upload
            if final_result == "Real":
                # Save post in the database
                new_post = Post(
                    user_id=session.get('user_id'),
                    file_path=file_path,
                    file_type=filename.rsplit('.', 1)[1].lower(),
                    is_deepfake=False,  # Because it's real
                    confidence=1.0  # Assume 100% confidence for real (can modify based on your logic)
                )
                db.session.add(new_post)
                db.session.commit()
                flash('File uploaded successfully!', 'success')
            else:
                # If the file is classified as "Fake," delete the file and reject the upload
                os.remove(file_path)
                flash('File is classified as fake and cannot be uploaded.', 'danger')
                return redirect(url_for('upload'))

            return redirect(url_for('result'))

    return render_template('upload.html')

@app.route('/logout')
def logout():
    # Clear the session and log the user out
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Get the user from the database
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        # Handle profile picture upload
        file = request.files.get('profile_pic')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            user.profile_pic = f'uploads/{filename}'
            db.session.commit()
            return redirect(url_for('profile'))
    return render_template('profile.html', user=user)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')

        # Update user details
        user.username = username
        user.email = email
        db.session.commit()

        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            reset_link = url_for('reset_password', _external=True) + f"?email={email}"
            return f'Reset link: <a href="{reset_link}">{reset_link}</a>'
        return 'Email not found!'
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.form.get('email')  # Get the email from form data
    new_password = request.form.get('new_password')  # Get the new password from form data

    if request.method == 'POST':
        if email and new_password:
            user = User.query.filter_by(email=email).first()
            if user:
                user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                db.session.commit()
                return 'Password has been updated!'
            else:
                return 'Invalid email address!'
        else:
            return 'Missing email or password!'
    elif request.method == 'GET':
        email = request.args.get('email')  # Get email from query parameters
        if email:
            return render_template('reset_password.html', email=email)
        else:
            return 'Invalid request!'


@app.route('/result')
def result():
    # Render the result page
    return render_template('result.html')


# 3 - Preprocess image for MesoNet
def preprocess_image_meso(image_path):
    img = load_img(image_path, target_size=(256, 256))  # Load and resize the image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

def preprocess_image_xception(image_path):
    img = load_img(image_path, target_size=(256, 256))  # Xception input size is 299x299
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# 4 - Predict function for MesoNet
def predict_meso(image_path):
    img_array = preprocess_image_meso(image_path)
    prediction = mesonet_model.predict(img_array)
    confidence = 1 - prediction[0][0] if prediction[0][0] < 0.5 else prediction[0][0]
    label = 'Real' if prediction[0][0] < 0.5 else 'Fake'
    return label, confidence , prediction[0][0]

def predict_xception(image_path):
    img_array = preprocess_image_xception(image_path)
    prediction = xceptionnet_model.predict(img_array)
    xception_out = prediction[0][0]
    confidence = 1 - prediction[0][0] if prediction[0][0] < 0.5 else prediction[0][0]
    label = 'Real' if prediction[0][0] < 0.5 else 'Fake'
    return label, confidence , xception_out

# 5 - Predict function for Vision Transformer
def predict_vit(image_path):
    image = Image.open(image_path).convert("RGB")
    image = processor_vit(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**image)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item()

    predicted_label = id2label[predicted_class_idx]
    if predicted_label=="Real":
        out=1-confidence
    else:
        out=confidence
    return predicted_label, confidence, out

# 6 - Image function to combine predictions and calculate dynamic weighted average
def image_classification_dynamic_weight(image_path):
    # Predict using MesoNet
    meso_label, meso_confidence , meso_out= predict_meso(image_path)
    print(f"MesoNet: {meso_label} with {meso_confidence * 100:.2f}% confidence")

    xception_label, xception_confidence,xception_out = predict_xception(image_path)
    print(f"XceptionNet: {xception_label} with {xception_confidence * 100:.2f}% confidence {xception_out} output ")

    # Predict using Vision Transformer
    vit_label, vit_confidence , vit_out= predict_vit(image_path)
    print(f"Vision Transformer: {vit_label} with {vit_confidence * 100:.2f}% confidence")
    weight_meso =0.25
    weight_vit = 0.5
    weight_xception = 0.25
    # Weighted average of confidences
    weighted_confidence = (weight_meso * meso_out) + (weight_vit * vit_out) + (weight_xception * xception_out )

    # Final prediction based on weighted confidence
    final_label = 'Real' if weighted_confidence < 0.5 else 'Fake'
    print(f"Final prediction based on weighted confidence: {final_label} with {weighted_confidence*100 }% confidence")
    print(f"Weight assigned to MesoNet: {weight_meso}, Weight assigned to Vision Transformer: {weight_vit}")

    return final_label, weighted_confidence


# Updated function to extract frames and save them to a local directory
def extract_frames(video_path, interval=30, output_folder='temp_frames'):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    # Create the folder for saving frames if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Capture frames at the given interval
        if frame_count % interval == 0:
            # Convert frame from BGR (used by OpenCV) to RGB (used by PIL and models)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # Save frame temporarily
            frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    return frames, output_folder



    
def video_classification(video_path,  clf=clf ,processor=processor_vit, vit_model= vit_model, mesonet_model= mesonet_model2, xcept_model= xcept_model2, interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    features_list = []
    predlist=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract a frame every 'interval' frames
        if frame_count % interval == 0:
            # Convert frame from BGR (OpenCV format) to RGB (PIL format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Extract features for the frame
            frame_features = extract_features_for_frame(frame_pil, processor, vit_model, mesonet_model, xcept_model)
            print([frame_features])
            predlist.append(clf.predict([frame_features])[0])

        frame_count += 1
    
    cap.release()
    
    # if len(features_list) == 0:
    #     raise ValueError("No frames extracted from the video.")
    
    # Aggregate features by averaging across all frames
    # avg_features = np.mean(np.array(features_list), axis=0).reshape(1, -1)
    
    # Predict using the ML model (random forest or any other)
    print(predlist)
    if predlist.count(0)>predlist.count(1):
        return 'Real'
    else:'Fake'
    
    # return "Real" if prediction[0] == 0 else "Fake"


# Video function to classify the entire video as real or fake
# def video_classification(video_path, interval=30):
#     # Extract frames from the video at the specified interval
#     frames, temp_folder = extract_frames(video_path, interval)

#     if len(frames) == 0:
#         raise ValueError("No frames extracted from video.")

#     total_confidence = 0
#     total_frames = len(frames)
#     fake_count = 0
#     real_count=0
#     # Loop over the frames and classify each one using the image classification function
#     for i, frame in enumerate(frames):
#         # Path to the frame image in the temp folder
#         temp_image_path = os.path.join(temp_folder, f'frame_{i * interval}.jpg')

#         # Classify the frame
#         result = image_classification_dynamic_weight(temp_image_path)
#         print(result)
#         # Determine if the frame is fake or real and calculate confidence
#         if result[0] == "Fake":
#             fake_count += 1
#         else:real_count+=1
#     print(fake_count,real_count)
#     fake_percentage = fake_count / total_frames
#     # Majority voting or confidence-based classification
#     if fake_count>=real_count:
#         final_video_label = "Fake"
#     else:
#         final_video_label = "Real"
#     print(f"The video is classified as: {final_video_label}")

#     # Clean up temporary files
#     for file_name in os.listdir(temp_folder):
#         file_path = os.path.join(temp_folder, file_name)
#         os.remove(file_path)
#     os.rmdir(temp_folder)

#     return final_video_label, fake_percentage

def extract_vit_features(image_path, processor_vit, model):
    if isinstance(image_path, Image.Image):
        image = image_path  # It's already an image object
    else:
        image = Image.open(image_path).convert("RGB")
    inputs = processor_vit(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[-1]  # Extract last hidden layer
        features = torch.mean(features, dim=1).numpy()  # Mean pooling
        return features.squeeze()

def toarray(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array
def predict_single_image(image_path):
    vit_features = extract_vit_features(image_path, processor_vit, vit_model)
    meso_features = mesonet_model2.predict(toarray(image_path))  # MesoNet features
    xcept_features = xcept_model2.predict(toarray(image_path))   # Xception features

    # Combine features or use them for further prediction
    combined_features = np.concatenate((vit_features, meso_features, xcept_features), axis=None)  # Flatten and concatenate

    return combined_features

def extract_features_for_frame(image, processor_vit, vit_model, mesonet_model, xcept_model):
    # Extract ViT features
    vit_features = extract_vit_features(image, processor_vit, vit_model)
    
    # Preprocess image for Keras models (MesoNet and Xception)
    img_array = np.expand_dims(np.array(image.resize((256, 256))) / 255.0, axis=0)  # Normalize
    
    # Extract MesoNet and Xception features
    meso_features = mesonet_model.predict(img_array)  # MesoNet features
    xcept_features = xcept_model.predict(img_array)  # Xception features
    
    # Combine features from ViT, MesoNet, and Xception
    combined_features = np.concatenate((vit_features, meso_features.flatten(), xcept_features.flatten()), axis=None)
    
    return combined_features

    

if __name__ == "__main__":
    app.run(debug=True, port=8000)

