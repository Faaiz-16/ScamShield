"""Lightweight language identification for English, Hindi and Hinglish."""

import re
from typing import Literal

MessageLanguage = Literal["en", "hi", "hinglish"]

# Frequent Hindi function words and verbs as they are typed in Roman script.
_HINGLISH_MARKERS = frozenset(
    """
    hai hain ho hoga hogi kar karo karein kare karna karke kiya kijiye aap aapka aapke
    aapki aapko apna apne apni mein se ka ki ke ko nahi nahin kya jayega jayegi jaoge
    gaye gaya gayi bhai abhi turant paise paisa bhej bhejo bata batao wala wali liye
    sirf yeh ye wo woh aaj kal baje raha rahe rahi hu hoon tum hum mera meri tera
    diya lena dena jaldi sab log kuch bahut warna toh bhi par pe agar lekin aur
    """.split()
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[a-z]+")


def detect_language(text: str) -> MessageLanguage:
    devanagari = len(_DEVANAGARI.findall(text))
    latin_words = _LATIN_WORD.findall(text.lower())
    latin_letters = sum(len(word) for word in latin_words)

    if devanagari and devanagari >= 0.5 * (devanagari + latin_letters):
        return "hi"

    if latin_words:
        hits = sum(word in _HINGLISH_MARKERS for word in latin_words)
        if hits >= 2 and hits / len(latin_words) >= 0.1:
            return "hinglish"
    return "en"
