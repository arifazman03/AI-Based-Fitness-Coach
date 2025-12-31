import os
import secrets
from PIL import Image
from flask import current_app

def save_picture(form_picture):
    """
    Saves the profile picture file to the static/profile_pics directory
    and resizes it to a square thumbnail.

    :param form_picture: The FileField data object from the Flask-WTF form.
    :return: The new random filename (string).
    """
    # 1. Generate a secure, random filename
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    
    # 2. Define the full path where the file will be saved
    # Ensure the path is correct for your 'static' folder structure
    upload_dir = os.path.join(current_app.root_path, 'static', 'profile_pics')
    picture_path = os.path.join(upload_dir, picture_fn)

    # 3. Resize the image (optional but recommended)
    output_size = (150, 150)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    
    # 4. Save the resized image
    i.save(picture_path)

    # 5. Return the filename to be saved in the database
    return picture_fn