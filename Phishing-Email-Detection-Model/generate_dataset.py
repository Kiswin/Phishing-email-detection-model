"""
generate_dataset.py
Creates a synthetic-but-realistic dataset of phishing and legitimate emails.
In a real submission you could swap this out for a public dataset
(e.g. the Kaggle "Phishing Email Detection" or "Nazario Phishing Corpus")
by loading a CSV with columns ['text', 'label'] instead.
"""

import random
import pandas as pd

random.seed(42)

# ---- Building blocks for PHISHING emails ----
phishing_subjects = [
    "Urgent: Your Account Will Be Suspended",
    "Action Required: Verify Your Identity Now",
    "Security Alert - Unusual Login Detected",
    "You Have Won a Prize! Claim Now",
    "Payment Failed - Update Your Billing Info",
    "Your Password Has Expired - Reset Immediately",
    "Final Notice: Account Deactivation Pending",
    "Confirm Your Bank Details to Avoid Suspension",
]

phishing_bodies = [
    "Dear Customer, we detected suspicious activity on your account. "
    "Click the link below within 24 hours to verify your identity or your account will be locked: {url}",

    "Your account has been temporarily suspended due to unusual activity. "
    "Please login immediately at {url} to restore access.",

    "Congratulations! You have been selected to receive a $500 gift card. "
    "Claim your reward now by visiting {url} before it expires.",

    "We were unable to process your last payment. Update your billing information "
    "urgently at {url} to avoid service interruption.",

    "This is an automated security notice. Someone tried to access your account from "
    "an unrecognized device. Verify it was you here: {url}",

    "Your mailbox has exceeded its storage limit. Click {url} now to upgrade your "
    "storage and avoid losing your emails permanently.",

    "IMPORTANT: Your tax refund of $1,240 is pending. Provide your bank details at "
    "{url} to receive the deposit immediately.",

    "We noticed a login attempt from a new location. If this wasn't you, secure "
    "your account right away at {url}.",
]

phishing_urls = [
    "http://192.168.44.12/secure-login",
    "http://paypa1-verify.com/account",
    "http://bank-secure-update.info/login",
    "http://amaz0n-rewards.net/claim",
    "http://appleid-confirm.tk/verify",
    "http://microsoft-support-alert.xyz",
    "http://192.34.55.100/login.php",
    "http://secure-update-account.click",
]

# ---- Building blocks for LEGITIMATE emails ----
legit_subjects = [
    "Meeting Reminder: Project Sync Tomorrow",
    "Your Monthly Newsletter",
    "Invoice #4521 for Your Recent Purchase",
    "Weekly Team Standup Notes",
    "Your Order Has Shipped",
    "Reminder: Dentist Appointment on Friday",
    "Welcome to Our Service!",
    "Your Flight Itinerary Confirmation",
]

legit_bodies = [
    "Hi team, just a reminder that our project sync is scheduled for tomorrow at 10am. "
    "Please review the attached agenda beforehand.",

    "Hello, thank you for subscribing to our newsletter. Here are this month's top "
    "articles and updates from our blog.",

    "Hi, your invoice for the recent purchase is attached. Total amount due is listed "
    "on the invoice. Let us know if you have any questions.",

    "Good morning everyone, here are the notes from this week's standup. Please add "
    "any comments directly in the shared doc.",

    "Hello, your order has shipped and is expected to arrive within 3-5 business days. "
    "You can track your package using the carrier's website.",

    "Hi, this is a friendly reminder about your dentist appointment scheduled for "
    "Friday at 2pm. Please call if you need to reschedule.",

    "Welcome aboard! We're excited to have you join our platform. Feel free to "
    "explore your dashboard and reach out if you have questions.",

    "Hello, attached is your flight itinerary for the upcoming trip. Please check "
    "in online 24 hours before departure.",
]

legit_urls = [
    "https://www.company.com/dashboard",
    "https://mail.google.com/inbox",
    "https://www.airline.com/itinerary",
    "https://www.yourbank.com/statements",
    "https://docs.company.com/notes",
    "https://www.onlinestore.com/orders",
    "https://www.dentistoffice.com/appointments",
    "https://www.newsletter.com/unsubscribe",
]

signoffs_legit = ["Best regards,\nSarah", "Thanks,\nThe Team", "Cheers,\nMike",
                  "Regards,\nCustomer Support", "Best,\nAlex"]


names = ["Alex", "Sarah", "Michael", "Priya", "John", "Emma", "Carlos", "Wei", "Fatima", "Liam"]
companies = ["Northbridge Bank", "Zenith Cloud", "Orbit Retail", "Vantage Systems",
             "Cascade Airlines", "Meridian Health", "Bluepeak Telecom", "Ivory Logistics"]
domains_legit = ["company.com", "orgmail.com", "workspace.io", "teamhub.net"]
domains_phish = ["secure-verify.info", "account-alert.click", "login-confirm.tk",
                  "support-update.xyz", "billing-check.click"]


def randomize(text: str) -> str:
    """Inject light lexical noise so the model can't just memorize exact templates."""
    text = text.replace("Customer", random.choice(names + ["Customer", "User"]))
    text = text.replace("$500", f"${random.choice([50, 120, 250, 500, 750, 999])}")
    text = text.replace("$1,240", f"${random.randint(200, 3000)}")
    if random.random() < 0.4:
        text += f"\n\nRef ID: {random.randint(100000, 999999)}"
    return text


def make_email(subject_pool, body_pool, url_pool, domain_pool, is_phishing):
    subject = random.choice(subject_pool)
    body_template = random.choice(body_pool)
    domain = random.choice(domain_pool)
    scheme = "http" if is_phishing and random.random() < 0.7 else "https"
    path = random.choice(["login", "verify", "account", "secure", "update", "confirm"])
    if is_phishing and random.random() < 0.3:
        url = f"http://{random.randint(10,199)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}/{path}"
    else:
        url = f"{scheme}://{random.choice(['www.', ''])}{domain}/{path}"

    body = body_template.format(url=url) if "{url}" in body_template else f"{body_template} {url}"
    body = randomize(body)

    if not is_phishing:
        body += "\n\n" + random.choice(signoffs_legit)
    else:
        footer = random.choice([
            "\n\nFailure to act within 24 hours will result in permanent suspension.",
            "\n\nThis link will expire soon. Act now to avoid losing access.",
            "\n\nOur security team requires immediate confirmation.",
            "",
        ])
        body += footer

    from_name = random.choice(companies)
    full_text = f"From: {from_name} <no-reply@{domain}>\nSubject: {subject}\n{body}"
    return full_text


def build_dataset(n_per_class=350):
    rows = []
    for _ in range(n_per_class):
        rows.append({
            "text": make_email(phishing_subjects, phishing_bodies, phishing_urls, domains_phish, True),
            "label": "phishing",
        })
    for _ in range(n_per_class):
        rows.append({
            "text": make_email(legit_subjects, legit_bodies, legit_urls, domains_legit, False),
            "label": "safe",
        })
    df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("emails_dataset.csv", index=False)
    print(f"Dataset created: {df.shape[0]} emails ({df['label'].value_counts().to_dict()})")
