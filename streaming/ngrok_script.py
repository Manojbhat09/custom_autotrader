
from pyngrok import ngrok
from login_app.app import app 

# Open a HTTP tunnel on the default port 80
public_url = ngrok.connect(port=5000, proto="http")
# Update the Flask app to use the public URL
print(public_url)
app.config['BASE_URL'] = public_url

# Now run the app
if __name__ == '__main__':
    app.run()