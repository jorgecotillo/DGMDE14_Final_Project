import smtplib

class EmailNotification(object):

    def send_email(self):
        # TODO: Use environment variables
        gmail_user = 'drowsiness.detection.no.reply@gmail.com'
        gmail_password = 'raspberrypi2021'

        sent_from = gmail_user
        to = ['jorge.cotillo@gmail.com']
        subject = 'OMG Super Important Message'
        body = "Hey, what's up?\n\n- You"

        email_text = """\
        From: %s
        To: %s
        Subject: %s

        %s
        """ % (sent_from, ", ".join(to), subject, body)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(gmail_user, gmail_password)
            server.sendmail(sent_from, to, email_text)
            server.close()

            print("Email sent!")

        except Exception as ex:
            print(ex)
            print("Something went wrong...")