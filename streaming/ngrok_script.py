
from pyngrok import ngrok
from login_app.app import app 

# Open a HTTP tunnel on the default port 80
public_url = ngrok.connect(port=5000)
import pdb; pdb.set_trace()
# Update the Flask app to use the public URL
app.config['BASE_URL'] = public_url

# Now run the app
if __name__ == '__main__':
    app.run()