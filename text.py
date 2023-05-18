"""Send a text message using Twilio."""
import os
from dotenv import load_dotenv

from twilio.rest import Client

load_dotenv()

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_twilio_message(message, to):
    """Send a message to a phone number."""
    message = client.messages.create(
        body=message,
        from_='+18578872406',
        to=to
    )

    return message.sid
