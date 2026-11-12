import { EXAMPLES, applyStaticText, initialLanguage, storeLanguage, translate } from "./i18n.js";

const MAX_CHARS = 5000;

const state = {
  language: "en",
  result: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);

const elements = {
  form: $("#check-form"),
  message: $("#message"),
  charCount: $("#char-count"),
  clear: $("#clear-btn"),
  submit: $("#submit-btn"),
  submitLabel: $("#submit-btn [data-i18n]"),
  formError: $("#form-error"),
  result: $("#result"),
  verdict: $("#verdict"),
  verdictLabel: $("#verdict-label"),
  scamType: $("#scam-type"),
  riskValue: $("#risk-value"),
  meterFill: $("#meter-fill"),
  confidence: $("#confidence"),
  reasons: $("#reasons"),
  actions: $("#actions"),
  linksSection: $("#links-section"),
  links: $("#links"),
  feedback: $("#feedback"),
  feedbackMain: $("#feedback-main"),
  feedbackLabels: $("#feedback-labels"),
  feedbackStatus: $("#feedback-status"),
};

function t(key, params) {
  return translate(state.language, key, params);
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function element(tag, { className, text } = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// --- Language ---------------------------------------------------------------

function applyLanguage(language) {
  state.language = language;
  applyStaticText(language);

  if (state.busy) elements.submitLabel.textContent = t("form.checking");
  if (state.result) renderResult(state.result);
}

// --- Form -------------------------------------------------------------------

function updateCharCount() {
  elements.charCount.textContent = `${elements.message.value.length} / ${MAX_CHARS}`;
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = !message;
}

function setBusy(busy) {
  state.busy = busy;
  elements.submit.disabled = busy;
  elements.submit.classList.toggle("is-loading", busy);
  elements.submitLabel.textContent = t(busy ? "form.checking" : "form.submit");
}

async function errorMessageFor(response) {
  if (response.status === 429) {
    const seconds = Number(response.headers.get("Retry-After")) || 30;
    return t("error.rateLimited", { seconds });
  }
  if (response.status === 422) {
    try {
      const body = await response.json();
      const field = body?.error?.details?.[0]?.field ?? "";
      if (field.endsWith("text")) return t("error.empty");
    } catch {
      // Fall through to the generic message.
    }
  }
  return t("error.server");
}

async function checkMessage(event) {
  event.preventDefault();
  if (state.busy) return;

  const text = elements.message.value.trim();
  if (!text) {
    showError(t("error.empty"));
    elements.message.focus();
    return;
  }
  if (text.length > MAX_CHARS) {
    showError(t("error.tooLong", { max: MAX_CHARS }));
    return;
  }

  showError("");
  setBusy(true);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language: state.language }),
    });
    if (!response.ok) {
      showError(await errorMessageFor(response));
      return;
    }
    state.result = await response.json();
    resetFeedback();
    renderResult(state.result);
    elements.result.hidden = false;
    elements.result.focus({ preventScroll: true });
    elements.result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    showError(t("error.network"));
  } finally {
    setBusy(false);
  }
}

// --- Result -----------------------------------------------------------------

function severityClass(severity) {
  if (severity >= 0.8) return "severity--high";
  if (severity >= 0.5) return "severity--medium";
  if (severity > 0) return "severity--low";
  return "severity--none";
}

function renderResult(result) {
  const texts = result.translations[state.language] ?? result.translations.en;
  const level = result.verdict;

  elements.verdict.className = `verdict verdict--${level}`;
  elements.result.className = `card result result--${level}`;
  elements.verdictLabel.textContent = texts.verdict_label;
  elements.scamType.textContent = texts.scam_type_label ?? "";
  elements.scamType.hidden = !texts.scam_type_label;

  elements.riskValue.textContent = percent(result.risk_score);
  elements.meterFill.style.width = percent(result.risk_score);
  elements.confidence.textContent = t("result.confidence", { value: percent(result.confidence) });

  elements.reasons.replaceChildren(
    ...result.reasons.map((reason, i) =>
      element("li", { className: severityClass(reason.severity), text: texts.reasons[i] }),
    ),
  );
  elements.actions.replaceChildren(...texts.actions.map((text) => element("li", { text })));

  renderLinks(result, texts);
  elements.feedback.hidden = !result.analysis_id;
}

function renderLinks(result, texts) {
  elements.linksSection.hidden = result.links.length === 0;
  const items = result.links.map((link, index) => {
    const item = element("li");
    // Shown as text only: a scam link must never become clickable here.
    item.append(element("span", { className: "link-list__domain", text: link.domain }));

    if (link.is_official) {
      const label = link.official_brand
        ? `${t("result.official")}: ${link.official_brand}`
        : t("result.official");
      item.append(element("span", { className: "badge badge--official", text: label }));
    } else if (link.findings.some((f) => f.severity >= 0.5)) {
      item.append(element("span", { className: "badge badge--danger", text: t("result.risky") }));
    }

    if (link.is_shortened && link.final_url !== link.url) {
      item.append(
        element("p", { className: "muted", text: t("result.goesTo", { url: link.final_url }) }),
      );
    }

    const findings = texts.link_findings[index] ?? [];
    if (findings.length) {
      const list = element("ul");
      list.append(...findings.map((text) => element("li", { text })));
      item.append(list);
    } else if (!link.is_official) {
      item.append(element("p", { className: "muted", text: t("result.noLinkIssues") }));
    }
    return item;
  });
  elements.links.replaceChildren(...items);
}

// --- Feedback ---------------------------------------------------------------

function resetFeedback() {
  elements.feedbackMain.hidden = false;
  elements.feedbackLabels.hidden = true;
  elements.feedbackStatus.textContent = "";
  elements.feedback.querySelectorAll("button").forEach((button) => {
    button.disabled = false;
  });
}

async function sendFeedback(isCorrect, correctLabel) {
  if (!state.result?.analysis_id) return;
  const buttons = elements.feedback.querySelectorAll("button");
  buttons.forEach((button) => {
    button.disabled = true;
  });

  const payload = { analysis_id: state.result.analysis_id, is_correct: isCorrect };
  if (correctLabel) payload.correct_label = correctLabel;

  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(String(response.status));
    elements.feedbackMain.hidden = true;
    elements.feedbackLabels.hidden = true;
    elements.feedbackStatus.textContent = t("feedback.thanks");
  } catch {
    elements.feedbackStatus.textContent = t("feedback.failed");
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

// --- Wiring -----------------------------------------------------------------

function init() {
  applyLanguage(initialLanguage());

  elements.form.addEventListener("submit", checkMessage);
  elements.message.addEventListener("input", () => {
    updateCharCount();
    if (elements.message.value.trim()) showError("");
  });
  elements.message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      elements.form.requestSubmit();
    }
  });

  elements.clear.addEventListener("click", () => {
    elements.message.value = "";
    updateCharCount();
    showError("");
    elements.message.focus();
  });

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.message.value = EXAMPLES[button.dataset.example];
      updateCharCount();
      showError("");
      elements.message.focus();
    });
  });

  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => {
      applyLanguage(button.dataset.lang);
      storeLanguage(button.dataset.lang);
    });
  });

  elements.feedbackMain.querySelector('[data-feedback="yes"]').addEventListener("click", () => {
    sendFeedback(true);
  });
  elements.feedbackMain.querySelector('[data-feedback="no"]').addEventListener("click", () => {
    elements.feedbackMain.hidden = true;
    elements.feedbackLabels.hidden = false;
    elements.feedbackLabels.querySelector("button").focus();
  });
  elements.feedbackLabels.querySelectorAll("[data-label]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(false, button.dataset.label || null));
  });

  updateCharCount();
}

init();
