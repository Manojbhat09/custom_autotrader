from flask import Flask, redirect, url_for, render_template, jsonify, request
from flask_dance.contrib.google import make_google_blueprint, google
import os
import jwt
import requests

app = Flask(__name__)
# Ideally, use environment variables for sensitive information
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "z5OzddOREf6qEZ62gosk8Y6b3uqHynpzlCrohaU3fe0")  # Replace with your secret key


@app.route('/google-login', methods=['POST'])
def google_login():
    import pdb; pdb.set_trace()
    # Extract the token sent by the client
    token = request.json.get('token')

    # Verify the token with Google's OAuth 2.0 server
    google_verify_url = 'https://oauth2.googleapis.com/tokeninfo'
    response = requests.get(google_verify_url, params={'id_token': token})
    user_info = response.json()

    # Check if the request was successful
    if user_info.get('aud') == '293772337596-dj5sf0svltplja3gs9qfcq694etrk6ug.apps.googleusercontent.com':
        # Here you can create a user session or perform actions for authenticated users
        return jsonify({'status': 'success', 'user': user_info})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid token'}), 401

# @app.route("/")
# def index():
#     if not google.authorized:
#         print("rendering the login template to login")
#         return render_template("login.html")
#     else:
#         print("user is logged in")
#     return


@app.route("/")
def index():
    return render_template("login.html")

if __name__ == '__main__':
    app.run(port=5000)