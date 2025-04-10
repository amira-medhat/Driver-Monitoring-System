import ipinfo
import smtplib
from email.message import EmailMessage

# === CONFIGURATION PARAMETERS ===
SENDER_EMAIL = "farahhahmed01@gmail.com"
APP_PASSWORD = "kjkskhjxietzmvfq"
RECEIVER_EMAIL = "farahhibrahim92@gmail.com"
DEFAULT_STREAM_URL = "https://www.youtube.com/watch?v=01Vv1YcmpzE"




IPINFO_TOKEN = "28ce483d77853b"  # <<< Paste your token here


# === ACCURATE IP-BASED LOCATION ===

def get_current_location():
    try:
        handler = ipinfo.getHandler(IPINFO_TOKEN)
        details = handler.getDetails()

        city = details.city or "Unknown city"
        country = details.country_name or "Unknown country"
        latlon = details.loc  # Format: "lat,lon"

        location = f"{city}, {country}"
        map_url = f"https://www.google.com/maps?q={latlon}" if latlon else "#"
        return location, map_url

    except Exception as e:
        print("⚠️ Failed to fetch location via IPInfo:", e)
        return "Location unavailable", "#"


# === EMAIL SENDER FUNCTION ===

def send_alert_email(url, to_email):
    location, map_url = get_current_location()

    msg = EmailMessage()
    msg['Subject'] = '🚨 Emergency Alert: Driver Unresponsive'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    # Fallback plain text
    msg.set_content(f'''
    ALERT! The driver may be asleep.
    Location: {location}
    Stream: {url}
    Map: {map_url}
    ''')

    html_content = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: 'Segoe UI', sans-serif;">
        <div style="max-width: 600px; margin: 40px auto; background: #ffffff; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); overflow: hidden;">

            <!-- Header -->
            <div style="background: linear-gradient(90deg, #00c9ff, #92fe9d); padding: 24px; text-align: center;">
                <h2 style="color: #003a53; margin: 0;">🚨 Driver Safety Alert</h2>
            </div>

            <!-- Body -->
            <div style="padding: 30px;">
                <p style="font-size: 17px; color: #333;">
                    <strong>Alert:</strong> The driver appears to be <span style="color: #ff4e50;"><strong>unresponsive</strong></span>.
                </p>
                <p style="font-size: 15px; color: #444;">
                    <strong>Last known location:</strong> <span style="color: #007bff;">{location}</span>
                </p>

                <!-- Live Location Button -->
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{map_url}" target="_blank"
                       style="background: linear-gradient(135deg, #28a745, #51e2a7); color: white; padding: 12px 24px; text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 30px;">
                       📍 View Live Location
                    </a>
                </div>

                <!-- Stream Button -->
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{url}" target="_blank"
                       style="background: linear-gradient(135deg, #007bff, #00c6ff); color: white; padding: 14px 28px; text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 30px;">
                       ▶️ View Live Stream
                    </a>
                </div>

                <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
                <p style="font-size: 13px; color: #999; text-align: center;">
                    Sent automatically by your Driver Monitoring System.<br>
                    This is an automated alert — please do not reply.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print("✅ Alert email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)


# # === TEST MAIN FUNCTION ===

# def main():
#     print("🚨 Sending test driver alert email with IPInfo location...")
#     send_alert_email(DEFAULT_STREAM_URL, RECEIVER_EMAIL)


# # === ENTRY POINT ===

# if __name__ == "__main__":
#     main()
