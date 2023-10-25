from flask import Flask, redirect, url_for
from flask_dance.contrib.google import make_google_blueprint, google
import jwt
app = Flask(__name__)
app.secret_key = 'random_secret_key'
blueprint = make_google_blueprint(client_id="293772337596-pb4g8lfra73eu2ila07j1gvsa469t06k.apps.googleusercontent.com", client_secret="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", redirect_to="https://aba1-49-43-26-171.ngrok-free.app/redirect_to_streamlit" )#"google_login")
app.register_blueprint(blueprint, url_prefix="/login")

@app.route("/redirect_to_streamlit")
def redirect_to_streamlit():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/plus/v1/people/me")
    assert resp.ok, resp.text
    email = resp.json()["emails"][0]["value"]
    # Create a token
    token = jwt.encode({'user': email}, app.secret_key, algorithm='HS256')
    # Redirect to Streamlit app with token
    return redirect(f"http://localhost:8501?token={token}")


@app.route('/')
def index():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/plus/v1/people/me")
    assert resp.ok, resp.text
    return "You are {email} on Google".format(email=resp.json()["emails"][0]["value"])


# from login_app.app import app 

# The URL you obtained from ngrok
# public_url = 'https://aba1-49-43-26-171.ngrok.io'
public_url= "https://aba1-49-43-26-171.ngrok-free.app"
# Update the Flask app to use the public URL
app.config['BASE_URL'] = public_url

if __name__ == '__main__':
    app.run()