"""User-facing bot text in English and Hindi. Values are Telegram HTML."""

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "start": (
            "<b>Welcome to ScamShield</b>\n\n"
            "Got a suspicious SMS, WhatsApp or UPI message? Forward it to me or paste it here, "
            "and I will tell you whether it looks like a scam and what to do.\n\n"
            "I understand English, Hindi and Hinglish.\n\n"
            "Use /language to switch between English and Hindi, and /help for more."
        ),
        "help": (
            "<b>How to use ScamShield</b>\n\n"
            "1. Forward the suspicious message to this chat, or copy and paste its text.\n"
            "2. I reply with a verdict (Safe, Suspicious or Likely Scam), the reasons, and what "
            "you should do.\n"
            "3. Tap <b>Correct</b> or <b>Wrong</b> to help me improve.\n\n"
            "<b>Privacy:</b> phone numbers, names and other personal details are removed before "
            "an anonymised copy is stored. Your Telegram ID is never stored.\n\n"
            "<b>Lost money?</b> Call the cyber crime helpline <b>1930</b> immediately or report "
            "at cybercrime.gov.in.\n\n"
            "Commands: /start, /help, /language"
        ),
        "choose_language": "Choose your language:",
        "language_set": "Language set to English.",
        "not_text": "Please send the message as text, or forward a message that contains text.",
        "too_long": "That message is too long. Please send at most {max} characters.",
        "rate_limited": "You are sending messages too quickly. Please wait {seconds} seconds.",
        "unavailable": "ScamShield is not reachable right now. Please try again in a few minutes.",
        "invalid": "I could not read that message. Please try sending it as plain text.",
        "expired": "This result has expired. Please send the message again.",
        "risk": "Risk",
        "why": "Why",
        "what_to_do": "What to do",
        "links": "Links in the message",
        "official": "official site",
        "risky": "risky",
        "no_link_issues": "no problems found",
        "disclaimer": "Automated check. It can be wrong, so stay careful.",
        "feedback_question": "Was this result correct?",
        "feedback_thanks": "Thank you for your feedback.",
        "feedback_failed": "Could not send feedback. Please try again.",
        "button_correct": "Correct",
        "button_wrong": "Wrong",
        "button_scam": "It was a scam",
        "button_legit": "It was genuine",
        "button_spam": "Just ads or spam",
        "button_unsure": "Not sure",
        "button_other_language": "हिंदी में देखें",
    },
    "hi": {
        "start": (
            "<b>ScamShield में आपका स्वागत है</b>\n\n"
            "कोई संदिग्ध एसएमएस, व्हाट्सएप या यूपीआई संदेश मिला है? उसे मुझे फॉरवर्ड करें या यहां "
            "चिपकाएं। मैं बताऊंगा कि यह धोखाधड़ी लगता है या नहीं और आपको क्या करना चाहिए।\n\n"
            "मैं अंग्रेज़ी, हिंदी और हिंग्लिश समझता हूं।\n\n"
            "भाषा बदलने के लिए /language और ज्यादा जानकारी के लिए /help भेजें।"
        ),
        "help": (
            "<b>ScamShield का उपयोग कैसे करें</b>\n\n"
            "1. संदिग्ध संदेश इस चैट में फॉरवर्ड करें, या उसका टेक्स्ट कॉपी करके चिपकाएं।\n"
            "2. मैं नतीजा (सुरक्षित, संदिग्ध या संभावित धोखाधड़ी), कारण और आपको क्या करना चाहिए, "
            "बताऊंगा।\n"
            "3. मुझे बेहतर बनाने के लिए <b>सही</b> या <b>गलत</b> दबाएं।\n\n"
            "<b>निजता:</b> गुमनाम कॉपी सेव करने से पहले फोन नंबर, नाम और अन्य निजी जानकारी हटा दी "
            "जाती है। आपकी टेलीग्राम आईडी कभी सेव नहीं होती।\n\n"
            "<b>पैसे कट गए?</b> तुरंत साइबर क्राइम हेल्पलाइन <b>1930</b> पर कॉल करें या "
            "cybercrime.gov.in पर शिकायत करें।\n\n"
            "कमांड: /start, /help, /language"
        ),
        "choose_language": "अपनी भाषा चुनें:",
        "language_set": "भाषा हिंदी कर दी गई है।",
        "not_text": "कृपया संदेश टेक्स्ट के रूप में भेजें, या ऐसा संदेश फॉरवर्ड करें जिसमें टेक्स्ट हो।",
        "too_long": "यह संदेश बहुत लंबा है। कृपया अधिकतम {max} अक्षर भेजें।",
        "rate_limited": "आप बहुत जल्दी-जल्दी संदेश भेज रहे हैं। कृपया {seconds} सेकंड रुकें।",
        "unavailable": "ScamShield से अभी संपर्क नहीं हो पा रहा। कृपया कुछ मिनट बाद फिर कोशिश करें।",
        "invalid": "मैं यह संदेश पढ़ नहीं सका। कृपया इसे सादे टेक्स्ट में भेजें।",
        "expired": "यह नतीजा अब उपलब्ध नहीं है। कृपया संदेश फिर से भेजें।",
        "risk": "जोखिम",
        "why": "क्यों",
        "what_to_do": "क्या करें",
        "links": "संदेश के लिंक",
        "official": "आधिकारिक साइट",
        "risky": "खतरनाक",
        "no_link_issues": "कोई समस्या नहीं मिली",
        "disclaimer": "यह स्वचालित जांच है और गलत हो सकती है, इसलिए सावधान रहें।",
        "feedback_question": "क्या यह नतीजा सही था?",
        "feedback_thanks": "आपकी प्रतिक्रिया के लिए धन्यवाद।",
        "feedback_failed": "प्रतिक्रिया नहीं भेजी जा सकी। कृपया फिर से कोशिश करें।",
        "button_correct": "सही",
        "button_wrong": "गलत",
        "button_scam": "यह धोखाधड़ी थी",
        "button_legit": "यह असली था",
        "button_spam": "सिर्फ विज्ञापन/स्पैम",
        "button_unsure": "पता नहीं",
        "button_other_language": "View in English",
    },
}


def text(language: str, key: str, **params: object) -> str:
    template = TEXTS.get(language, TEXTS["en"]).get(key) or TEXTS["en"][key]
    return template.format(**params) if params else template
