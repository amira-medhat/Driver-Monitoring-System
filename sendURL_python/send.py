"""
@file send.py
@brief Sends a modern styled HTML emergency alert email with a live stream URL when the driver is unresponsive.
"""

import smtplib
from email.message import EmailMessage
from config import SENDER_EMAIL, APP_PASSWORD, RECEIVER_EMAIL, ALERT_URL



# === FUNCTION DEFINITIONS ===

def send_alert_email(url, to_email):
    """
    Sends a modern-styled HTML email alert with the given live stream URL.
    """
    msg = EmailMessage()
    msg['Subject'] = '🚨 Emergency Alert: Driver Unresponsive'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    msg.set_content(f'''
    ALERT! The driver may be asleep. Please check the situation immediately.
    Link: {url}
    ''')

    html_content = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #e9ecef; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <div style="max-width: 600px; margin: 40px auto; background: #ffffff; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); overflow: hidden;">

            <!-- Glowing Header -->
            <div style="background: linear-gradient(90deg, #00c9ff, #92fe9d); padding: 24px; text-align: center; box-shadow: 0 0 12px rgba(0, 201, 255, 0.4);">
                <h2 style="color: #003a53; margin: 0; font-size: 24px;">🚨 Driver Safety Alert 🚨</h2>
            </div>


            <!-- Body -->
            <div style="padding: 30px;">
                <p style="font-size: 17px; color: #333;">
                    <strong>Alert:</strong> The driver appears to be <span style="color: #ff4e50;"><strong>unresponsive</strong></span>. Immediate attention is required.
                </p>
                <p style="font-size: 15px; color: #555; line-height: 1.6;">
                    Please attempt to contact the driver. If you can't reach them, access the live stream feed below to verify their condition.
                </p>

                <!-- Button -->
                <div style="text-align: center; margin: 35px 0;">
                    <a href="{url}" target="_blank" 
                        style="background: linear-gradient(135deg, #007bff, #00c6ff); color: white; padding: 14px 28px; text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 30px; box-shadow: 0 4px 15px rgba(0,123,255,0.4); transition: background 0.3s;">
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
            print("✅ alert email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# === MAIN TEST ===

if __name__ == "__main__":
    send_alert_email(ALERT_URL, RECEIVER_EMAIL)
