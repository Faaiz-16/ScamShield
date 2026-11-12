# Labeling Guide

This guide defines how messages are collected, formatted and labelled for
training and evaluating the ScamShield classifiers.

> **About `seed_sample.csv`:** every row in that file is a synthetic sample
> written for development and testing (`source = synthetic_sample`). Phone
> numbers, names and amounts in it are fictional. It is far too small and too
> clean to measure real-world accuracy. Replace or extend it with collected
> data before reporting results.

## File format

UTF-8 CSV with a header row and these columns, in this order:

| Column      | Required | Description |
|-------------|----------|-------------|
| `id`        | yes      | Unique, stable identifier, e.g. `sms-000123`. Never reuse an id. |
| `text`      | yes      | The full message text exactly as received (line breaks allowed inside quotes). |
| `language`  | yes      | `en`, `hi` or `hinglish` (see below). |
| `label`     | yes      | `scam`, `legit` or `spam_not_scam` (see below). |
| `source`    | yes      | Where the message came from: `sms`, `whatsapp`, `telegram`, `email`, `public_report`, `synthetic_sample`, ... |
| `scam_type` | for scams| One category from the list below. Must be empty for `legit` and `spam_not_scam`. |

Example:

```csv
id,text,language,label,source,scam_type
sms-000001,"Your KYC has expired. Update now: http://sbi-kyc.xyz",en,scam,sms,kyc_scam
sms-000002,"482913 is your OTP for login. Do not share it with anyone.",en,legit,sms,
```

## Labels

### `scam`
The message tries to make the recipient lose money, credentials or control of
an account or device through deception. Label as `scam` if **any** of these hold:

- It asks for an OTP, PIN, CVV, password or card details.
- It asks the user to enter a UPI PIN or scan a QR code to *receive* money.
- It impersonates a bank, government body, police, courier, utility or company
  and asks the user to click a link, call a personal number, install an app or pay.
- It demands an upfront fee for a job, loan, prize, refund or parcel release.
- It promises guaranteed or unrealistic returns.
- It links to a lookalike domain of a real brand (e.g. `hdfcbannk.com`).

### `legit`
A genuine message that a real organisation or person would send. Typical examples:

- Bank OTPs, debit/credit alerts and balance updates.
- Delivery, booking and appointment updates.
- Bill reminders that point to the official app or website.
- Personal messages from friends and family.

Genuine bank messages often contain the same words as scams ("OTP", "blocked",
"KYC"). Label these carefully: they are the main source of false positives.

### `spam_not_scam`
Unsolicited promotional or bulk messages that are annoying but not fraudulent:
sales, offers, coaching classes, real estate, legitimate pre-approved credit
card offers that point to official channels.

### Edge cases

| Situation | Label |
|-----------|-------|
| Genuine bank OTP that says "do not share" | `legit` |
| Promotional offer with a link to the brand's real domain | `spam_not_scam` |
| Promotional offer with a link to a lookalike domain | `scam` |
| "Your card is blocked as per your request" from the bank's sender ID | `legit` |
| "Your account will be blocked, update KYC at <link>" | `scam` |
| Unsure, and the message cannot be verified | Skip it. Do not guess. |

## Scam types

Use exactly one of these values in `scam_type` (they match the rule engine
categories in `app/detection/rules.yaml`). Pick the one that best describes
the scammer's main goal. Use `other` only if nothing fits.

| Value | Meaning |
|-------|---------|
| `credential_theft` | Asks for OTP, PIN, CVV, password or card/account details |
| `kyc_scam` | Fake KYC, PAN or Aadhaar update |
| `account_threat` | Threat to block an account without another clear scheme |
| `upi_fraud` | UPI collect requests, "PIN to receive", QR scams, wrong-transfer tricks |
| `utility_scam` | Electricity, gas or water disconnection threats |
| `job_scam` | Fake jobs, task-based earning, registration fees |
| `lottery_scam` | Lottery, KBC, lucky draw, prize claims |
| `customer_care_scam` | Fake customer care or helpline numbers |
| `remote_access_scam` | Asks to install AnyDesk, TeamViewer or similar |
| `malware` | Asks to install an APK or app from a link |
| `parcel_scam` | Parcel held, address update or customs fee |
| `impersonation_scam` | Police, CBI, customs, "digital arrest" threats |
| `tax_refund_scam` | Fake income tax refunds |
| `reward_scam` | Expiring reward points |
| `telecom_scam` | SIM block, eSIM swap, TRAI threats |
| `investment_scam` | Guaranteed returns, trading or crypto groups |
| `loan_scam` | Instant loans with upfront fees |
| `challan_scam` | Fake traffic challans |
| `freebie_scam` | Fake government schemes or free gifts |
| `phishing_link` | Main sign is a fake or lookalike website link |
| `other` | None of the above |

## Language

- `hi`: written mainly in Devanagari script.
- `hinglish`: Hindi written in Roman script, with or without English words
  ("Aapka account band ho jayega").
- `en`: English, even if it contains a few Indian names or brand words.

Mixed Devanagari and Roman text: use the script that carries most of the
sentence. `app/detection/language.py` gives a suggestion, but the human label wins.

## Privacy and anonymisation

- Collect messages only with the consent of the person who received them.
- Keep raw collected files in `data/raw/`. This folder is git-ignored and must
  never be committed or shared.
- Run `python -m ml.prepare_data` before training. It replaces phone numbers,
  emails, UPI IDs, card, Aadhaar and PAN numbers, account numbers, dates,
  amounts, OTPs and names after a salutation with placeholders such as
  `<PHONE>` and `<AMOUNT>`. Link paths are removed; link domains are kept.
- Name detection is heuristic. Before sharing a processed file, skim it for
  names that were missed (e.g. "delivery partner Ravi") and remove them.

## Quality checks

1. Label each message independently of its source; do not assume all messages
   from a "bank" sender ID are genuine.
2. Have a second person label a random 10% sample and compare. Discuss and
   resolve disagreements, then update this guide if a rule was unclear.
3. Keep the classes reasonably balanced, and make sure each language has
   examples of all three labels.
4. `ml.prepare_data` rejects files with missing columns, unknown labels,
   duplicate ids, or the same text labelled differently.
