import os
import random
import string
import re
import json
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, g
from dotenv import load_dotenv
from datetime import datetime, timezone
from functools import wraps

import google.generativeai as genai

# --- Firebase ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
# We need 'auth' now ^

# --- Stability AI ---
import stability_sdk.client as StabilityClient
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# --- Cloudinary ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- VERCEL FIX 1: Load Environment Variables ---
load_dotenv() 

# --- VERCEL FIX 2: Handle serviceAccountKey.json ---
service_account_json_string = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

try:
    if service_account_json_string:
        service_account_info = json.loads(service_account_json_string)
        cred = credentials.Certificate(service_account_info)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    
    # Check if app is already initialized to avoid crashing on Vercel's hot-reloads
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except FileNotFoundError:
    print("="*50)
    print("ERROR: serviceAccountKey.json not found (for local dev).")
    print("="*50)
except json.JSONDecodeError:
    print("="*50)
    print("ERROR: FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON.")
    print("="*50)

db = firestore.client()
# --- END OF VERCEL FIXES ---


# --- VERCEL FIX 3: Tell Flask where the 'templates' folder is ---
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates')
           )
# --- END OF VERCEL FIX ---

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_strong_default_secret_key_12345')

# --- Configure Gemini ---
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Set it in Vercel Environment Variables.")
genai.configure(api_key=api_key) 
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Configure Stability AI ---
stability_api_key = os.environ.get("STABILITY_API_KEY")
if not stability_api_key:
    raise ValueError("STABILITY_API_KEY not found. Set it in Vercel Environment Variables.")
stability_api = StabilityClient.StabilityInference(
    key=stability_api_key,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

# --- Configure Cloudinary ---
cloudinary.config( 
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.environ.get("CLOUDINARY_API_KEY"), 
    api_secret = os.environ.get("CLOUDINARY_API_SECRET") 
)
if not os.environ.get("CLOUDINARY_CLOUD_NAME"):
    raise ValueError("CLOUDINARY_CLOUD_NAME not found. Set it in Vercel Environment Variables.")


# === NEW: Authentication Decorator ===
# This is a "wrapper" that we'll add to routes we want to protect
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            # If called from a server-side render, check session
            id_token = session.get('id_token')
            if not id_token:
                return redirect(url_for('login'))

        # Clean up the "Bearer " prefix if it exists
        if id_token.startswith('Bearer '):
            id_token = id_token.split(' ')[1]
            
        try:
            # Verify the token with Firebase Admin
            decoded_token = auth.verify_id_token(id_token)
            # Store the user's info in a global 'g' object for this request
            g.user = decoded_token
            g.user_uid = decoded_token['uid']
        except auth.InvalidIdTokenError:
            # Token is invalid or expired
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Token verification error: {e}")
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

# --- Helper Functions (Unchanged) ---
def generate_room_code(length=4):
    while True:
        code = ''.join(random.choices(string.digits, k=length))
        room_ref = db.collection('rooms').document(code)
        if not room_ref.get().exists:
            return code

# We don't need get_user_id() or get_nickname() anymore.
# We will use g.user_uid and g.user['email']

# --- build_gemini_prompt (Updated) ---
def build_gemini_prompt(room_data, user_nickname, message_text):
    prompt_lines = [
        f"You are {room_data['bot_name']}. Your personality is: {room_data['bot_personality']}.",
        f"Your appearance is: {room_data.get('bot_appearance', 'not specified')}",
        "You are in a role-playing game where multiple users are trying to win your affection.",
    ]
    prompt_lines.append("\nCurrent affection levels:")
    if not room_data.get('users'):
        prompt_lines.append("No one is in the room yet.")
    else:
        for user_id, data in room_data['users'].items():
            prompt_lines.append(f"- {data['nickname']}: {data['score']}%")

    prompt_lines.append(f"\nThe current scenario: {room_data['start_scenario']}")
    prompt_lines.append("\nHere is the recent chat history (max 10):")
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    
    for msg in sorted_messages[-10:]:
        sender = msg['user_id'] # Changed from 'user'
        if sender == room_data['bot_name']: # This is still fine
            sender_display = "You"
        elif sender == "System":
            sender_display = "System"
        else:
            sender_display = next((data['nickname'] for uid, data in room_data.get('users', {}).items() if uid == sender), sender)
        
        prompt_lines.append(f"{sender_display}: {msg['text']}")

    prompt_lines.append(f"\n--- NEW MESSAGE ---")
    prompt_lines.append(f"{user_nickname}: {message_text}")
    prompt_lines.append("\n--- YOUR TASK ---")
    prompt_lines.append(
        "Based on this new message, you must do two things:"
        f"\n1.  **Respond** in character as {room_data['bot_name']}."
        "\n2.  **Evaluate** the user's message. How much did it change your affection for them?"
        f" The difficulty is {room_data['difficulty']}/10."
        "\nYou MUST reply in this exact JSON format (no markdown):"
        "\n{"
        "\n  \"response\": \"Your in-character reply here.\","
        "\n  \"affection_change\": <number from -20 to 20>"
        "\n}"
    )
    return "\n".join(prompt_lines)

# --- parse_gemini_response (Unchanged) ---
def parse_gemini_response(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print(f"Parsing Error: No JSON block found in response. Response was: {text}")
            return {"response": text, "affection_change": 0}
        json_text = match.group(0)
        data = json.loads(json_text)
        return {
            "response": data.get("response", "I'm not sure what to say."),
            "affection_change": int(data.get("affection_change", 0))
        }
    except Exception as e:
        print(f"Error parsing Gemini response: {e} - Response was: {text}")
        return {"response": text, "affection_change": 0}

# --- index route (Updated) ---
@app.route("/")
def index():
    # This is now the public bots page
    return redirect(url_for('public_bots'))

# --- === NEW AUTH ROUTES === ---
@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/set-token", methods=["POST"])
def set_token():
    """Receives ID token from client-side JS and stores it in the session."""
    try:
        data = request.get_json()
        id_token = data['token']
        # Verify it once before storing
        auth.verify_id_token(id_token)
        # Store the token in a secure, HttpOnly session cookie
        session['id_token'] = id_token
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error setting token: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/logout")
def logout():
    session.pop('id_token', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

# --- === NEW FEATURE ROUTES === ---

@app.route("/public-bots")
def public_bots():
    bots = []
    try:
        # Query Firestore for all bots that are public
        bots_ref = db.collection('rooms').where('is_public', '==', True).limit(50)
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            bot_data['id'] = doc.id # Add the room_code as 'id'
            bots.append(bot_data)
    except Exception as e:
        print(f"Error fetching public bots: {e}")
        flash(f"Error fetching bots: {e}")
        
    return render_template("public_bots.html", bots=bots)

@app.route("/my-bots")
@login_required # Protect this route
def my_bots():
    bots = []
    user_uid = g.user_uid # Get the user's ID from the decorator
    try:
        # Query Firestore for bots where owner_uid matches
        bots_ref = db.collection('rooms').where('owner_uid', '==', user_uid).limit(50)
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            bot_data['id'] = doc.id
            bots.append(bot_data)
    except Exception as e:
        print(f"Error fetching user's bots: {e}")
        flash(f"Error fetching your bots: {e}")
        
    return render_template("my_bots.html", bots=bots)

# --- /generate-bot-image (Now login protected) ---
@app.route("/generate-bot-image", methods=["POST"])
@login_required # Only logged-in users can generate images
def generate_bot_image():
    # ... (function content is exactly the same as before) ...
    try:
        data = request.get_json()
        gender = data.get('gender', 'person')
        age = data.get('age', '20')
        appearance = data.get('appearance', 'average')
        prompt = f"A beautiful portrait of a {age} year old {gender}, {appearance}. digital art, anime style, detailed face, cinematic lighting, high quality"
        negative_prompt = "blurry, deformed, ugly, bad anatomy, mutated, extra limbs, disfigured"
        print(f"Stability Prompt: {prompt}")
        answers = stability_api.generate(
            prompt=[
                generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=1.0)),
                generation.Prompt(text=negative_prompt, parameters=generation.PromptParameters(weight=-1.0))
            ],
            style_preset="anime", steps=30, cfg_scale=7.0, width=512, height=512, samples=1
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    return jsonify({'success': False, 'error': 'Image generation was filtered for safety. Try a different prompt.'}), 400
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img_data = artifact.binary
                    print("Uploading to Cloudinary...")
                    upload_result = cloudinary.uploader.upload(
                        img_data, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}"
                    )
                    secure_url = upload_result.get('secure_url')
                    if not secure_url:
                        return jsonify({'success': False, 'error': 'Failed to upload image to Cloudinary.'}), 500
                    print(f"Image uploaded: {secure_url}")
                    return jsonify({'success': True, 'image_url': secure_url})
        return jsonify({'success': False, 'error': 'No image was generated.'}), 500
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- /upload-bot-image (Now login protected) ---
@app.route("/upload-bot-image", methods=["POST"])
@login_required # Only logged-in users can upload images
def upload_bot_image():
    # ... (function content is exactly the same as before) ...
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    if file:
        try:
            print("Uploading custom image to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                file, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}"
            )
            secure_url = upload_result.get('secure_url')
            if not secure_url:
                return jsonify({'success': False, 'error': 'Failed to upload to Cloudinary.'}), 500
            print(f"Image uploaded: {secure_url}")
            return jsonify({'success': True, 'image_url': secure_url})
        except Exception as e:
            print(f"Error uploading image: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Unknown file error'}), 500


# --- create_room (Updated) ---
@app.route("/create", methods=["GET", "POST"])
@login_required # Protect this route
def create_room():
    if request.method == "POST":
        room_code = generate_room_code()
        user_uid = g.user_uid # Get the logged-in user's ID
        user_email = g.user.get('email', 'User') # Get their email
        
        bot_name = request.form.get('bot_name', 'Bot')
        start_scenario = request.form.get('start_scenario', 'You meet at a park.')
        bot_image_url = request.form.get('bot_image_url') or "https://placehold.co/100x100/4a5568/FFFFFF?text=Bot"
        bot_appearance = request.form.get('appearance', 'not specified')
        
        # --- NEW: Get public/private state ---
        is_public = request.form.get('is_public') == 'on' # Checkbox value

        py_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc) 

        new_room_data = {
            'bot_name': bot_name,
            'bot_personality': request.form.get('bot_personality', 'friendly'),
            'start_scenario': start_scenario,
            'difficulty': int(request.form.get('difficulty', 5)),
            'game_over': False,
            'users': {}, # A room is empty until someone joins
            'messages': [
                 {
                    'user_id': 'System', 
                    'text': f"Game started! {user_email} created the room.",
                    'timestamp': py_timestamp 
                },
                {
                    'user_id': bot_name, # Bot's message
                    'text': start_scenario,
                    'timestamp': py_timestamp 
                }
            ],
            'bot_image_url': bot_image_url,
            'bot_appearance': bot_appearance,
            # --- NEW: Ownership and Public fields ---
            'owner_uid': user_uid,
            'owner_email': user_email,
            'is_public': is_public
        }
        
        db.collection('rooms').document(room_code).set(new_room_data)
        
        # Redirect to the 'My Bots' page instead of the room
        return redirect(url_for('my_bots'))
    
    return render_template("create.html")

# --- join_room (Updated) ---
@app.route("/join", methods=["GET", "POST"])
@login_required # User must be logged in to join any room
def join_room():
    room_code = request.args.get('code') # Get from URL parameter
    
    if request.method == 'POST':
        # This handles the "Join by Code" form
        room_code = request.form.get('room_code')

    if not room_code:
        flash("No room code provided.")
        return redirect(url_for('index'))

    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Error: Room code not found.")
        return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
        
    if room_data.get('game_over', False):
        flash("Error: This game has already ended.")
        return redirect(url_for('index'))
    
    # Check if bot is private AND user is not the owner
    if not room_data.get('is_public', False) and room_data.get('owner_uid') != g.user_uid:
        # This is a private bot. We must join by code.
        # This logic is fine, as the only way here is if they POSTed a code
        pass

    user_uid = g.user_uid
    nickname = g.user.get('email', f"User_{random.randint(100, 999)}")
    
    # Add user to room if not already in it
    if user_uid not in room_data.get('users', {}):
        py_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        room_ref.update({
            f'users.{user_uid}': {'nickname': nickname, 'score': 0},
            'messages': firestore.ArrayUnion([
                {
                    'user_id': 'System',
                    'text': f"{nickname} has joined the game!",
                    'timestamp': py_timestamp 
                }
            ])
        })
        
    # User is now in the room, send them to the chat
    return redirect(url_for('chat_room', room_code=room_code))

# --- chat_room (Updated) ---
@app.route("/room/<room_code>", methods=["GET", "POST"])
@login_required # Must be logged in to chat
def chat_room(room_code):
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Error: Room not found.")
        return redirect(url_for('index'))
        
    user_uid = g.user_uid
    room_data = room_doc.to_dict()
    
    # Get user's nickname from the 'users' dict in the room
    # This ensures they have properly "joined"
    user_data = room_data.get('users', {}).get(user_uid)
    if not user_data:
        # User is not in this room. Kick them to the join page.
        flash("You have not joined this room. Joining now...")
        return redirect(url_for('join_room', code=room_code))
    
    nickname = user_data.get('nickname', 'Player')

    # Handle a new message submission
    if request.method == "POST":
        if room_data.get('game_over', False):
            return jsonify({'success': False, 'error': 'Game is over'}), 400

        message_text = request.form.get('message')
        if not message_text:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
            
        try:
            py_timestamp_start = datetime.utcnow().replace(tzinfo=timezone.utc)

            prompt = build_gemini_prompt(room_data, nickname, message_text)
            response = model.generate_content(prompt)
            parsed_data = parse_gemini_response(response.text)
            
            bot_response = parsed_data['response']
            affection_change = parsed_data['affection_change']

            py_timestamp_end = datetime.utcnow().replace(tzinfo=timezone.utc)

            new_score = user_data['score'] + affection_change
            new_score = max(0, min(100, new_score))
            
            score_msg = f"{nickname}'s affection didn't change. (Still {new_score}%)"
            if affection_change > 0:
                score_msg = f"{nickname}'s affection went up by {affection_change}%! (Now {new_score}%)"
            elif affection_change < 0:
                score_msg = f"{nickname}'s affection went down by {abs(affection_change)}%! (Now {new_score}%)"

            messages_to_add = [
                {'user_id': user_uid, 'text': message_text, 'timestamp': py_timestamp_start},
                {'user_id': room_data['bot_name'], 'text': bot_response, 'timestamp': py_timestamp_end},
                {'user_id': 'System', 'text': score_msg, 'timestamp': py_timestamp_end}
            ]

            update_data = {
                f'users.{user_uid}.score': new_score
            }

            if new_score >= 100:
                update_data['game_over'] = True
                messages_to_add.append({
                    'user_id': 'System',
                    'text': f"GAME OVER! {nickname} has won {room_data['bot_name']}'s affection!",
                    'timestamp': py_timestamp_end
                })
            
            update_data['messages'] = firestore.ArrayUnion(messages_to_add)
            room_ref.update(update_data)
            
            return jsonify({'success': True})

        except Exception as e:
            print(f"Error during Gemini call: {e}")
            py_timestamp_error = datetime.utcnow().replace(tzinfo=timezone.utc)
            room_ref.update({
                'messages': firestore.ArrayUnion([
                    {'user_id': user_uid, 'text': message_text, 'timestamp': py_timestamp_error},
                    {'user_id': 'System', 'text': f"Sorry, {nickname}, I'm having trouble thinking. (Error: {e})", 'timestamp': py_timestamp_error}
                ])
            })
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Check if the logged-in user is the owner of this bot
    is_owner = (room_data.get('owner_uid') == user_uid)

    return render_template("room.html", room=room_data, room_code=room_code, user_id=user_uid, is_owner=is_owner)


# --- === NEW DELETE BOT ROUTE === ---
@app.route("/delete-bot", methods=["POST"])
@login_required
def delete_bot():
    room_code = request.form.get('room_code')
    user_uid = g.user_uid

    if not room_code:
        flash("No room code provided.")
        return redirect(url_for('my_bots'))
    
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Room not found.")
        return redirect(url_for('my_bots'))

    room_data = room_doc.to_dict()

    # CRITICAL: Check if the logged-in user is the owner
    if room_data.get('owner_uid') != user_uid:
        flash("You do not have permission to delete this bot.")
        return redirect(url_for('my_bots'))
    
    try:
        # 1. Delete the bot from Firestore
        room_ref.delete()
        
        # 2. (Optional but good) Delete the image from Cloudinary
        image_url = room_data.get('bot_image_url')
        if image_url and 'cloudinary.com' in image_url:
            # Extract the public_id from the URL
            # e.g., .../upload/v12345/bot_avatars/bot_abcdef.png
            public_id_match = re.search(r'bot_avatars/([^.]+)', image_url)
            if public_id_match:
                public_id = f"bot_avatars/{public_id_match.group(1)}"
                print(f"Deleting image from Cloudinary: {public_id}")
                cloudinary.uploader.destroy(public_id)

        flash(f"Bot '{room_data['bot_name']}' has been deleted.")
        
    except Exception as e:
        print(f"Error deleting bot: {e}")
        flash(f"An error occurred while deleting the bot: {e}")

    return redirect(url_for('my_bots'))
