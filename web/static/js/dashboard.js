import { applyStaticText, initialLanguage, storeLanguage, translate } from "./i18n.js";

const VERDICTS = ["safe", "suspicious", "likely_scam"];

const state = {
  language: "en",
  days: 30,
  stats: null,
  charts: {},
};

const $ = (selector) => document.querySelector(selector);

function t(key, params) {
  return translate(state.language, key, params);
}

function locale() {
  return state.language === "hi" ? "hi-IN" : "en-IN";
}

function formatNumber(value) {
  return new Intl.NumberFormat(locale()).format(value);
}

function share(part, total) {
  return total ? `${Math.round((part / total) * 100)}%` : "0%";
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function cell(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function tableRows(tbody, rows) {
  tbody.replaceChildren(
    ...rows.map((values) => {
      const row = document.createElement("tr");
      row.append(...values.map((value, i) => cell(i === 0 ? "th" : "td", String(value))));
      row.firstChild.scope = "row";
      return row;
    }),
  );
}

/** One entry per calendar day (UTC, like the API), with zero for days without checks. */
function fillDays(daily, days) {
  const byDate = new Map(daily.map((entry) => [entry.date, entry]));
  const today = new Date();
  const result = [];
  for (let offset = days - 1; offset >= 0; offset -= 1) {
    const day = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate() - offset));
    const iso = day.toISOString().slice(0, 10);
    result.push(byDate.get(iso) ?? { date: iso, total: 0, likely_scam: 0 });
  }
  return result;
}

function formatDay(iso) {
  return new Intl.DateTimeFormat(locale(), { day: "numeric", month: "short", timeZone: "UTC" }).format(
    new Date(`${iso}T00:00:00Z`),
  );
}

// --- Rendering ---------------------------------------------------------------

function renderTiles(stats) {
  const { total, verdicts, feedback } = stats;
  $("#stat-total").textContent = formatNumber(total);
  $("#stat-range").textContent = t("dash.days", { n: stats.days });
  $("#stat-scams").textContent = formatNumber(verdicts.likely_scam);
  $("#stat-scams-share").textContent = t("dash.share", { value: share(verdicts.likely_scam, total) });
  $("#stat-suspicious").textContent = formatNumber(verdicts.suspicious);
  $("#stat-suspicious-share").textContent = t("dash.share", {
    value: share(verdicts.suspicious, total),
  });

  const rated = feedback.correct + feedback.wrong;
  if (rated) {
    $("#stat-feedback").textContent = share(feedback.correct, rated);
    $("#stat-feedback-detail").textContent = t("dash.feedbackDetail", {
      correct: formatNumber(feedback.correct),
      rated: formatNumber(rated),
    });
  } else {
    $("#stat-feedback").textContent = "-";
    $("#stat-feedback-detail").textContent = t("dash.feedbackNone");
  }
}

/** Split a label into lines of at most maxChars, so long labels fit narrow screens. */
function wrapLabel(text, maxChars) {
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    if (line && (line + " " + word).length > maxChars) {
      lines.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  }
  if (line) lines.push(line);
  return lines.length > 1 ? lines : text;
}

function chartTheme() {
  return {
    text: cssVar("--text-muted"),
    grid: cssVar("--border"),
    brand: cssVar("--brand"),
    safe: cssVar("--safe"),
    warn: cssVar("--warn"),
    danger: cssVar("--danger"),
    surface: cssVar("--surface"),
  };
}

function drawChart(key, canvas, config) {
  state.charts[key]?.destroy();
  state.charts[key] = new window.Chart(canvas, config);
}

function renderTypes(types, theme) {
  $("#types-empty").hidden = types.length > 0;
  tableRows(
    $("#types-table"),
    types.map((type) => [type.label, formatNumber(type.count)]),
  );

  const container = $(".chart--bars");
  container.hidden = types.length === 0;
  if (!window.Chart || types.length === 0) return;

  const narrow = window.matchMedia("(max-width: 560px)").matches;
  container.style.height = `${Math.max(160, types.length * (narrow ? 46 : 40) + 40)}px`;
  drawChart("types", $("#types-chart"), {
    type: "bar",
    data: {
      labels: types.map((type) => wrapLabel(type.label, narrow ? 16 : 40)),
      datasets: [{ data: types.map((type) => type.count), backgroundColor: theme.danger, borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (items) => types[items[0].dataIndex].label } },
      },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0, color: theme.text }, grid: { color: theme.grid } },
        y: { ticks: { color: theme.text, autoSkip: false }, grid: { display: false } },
      },
    },
  });
}

function renderVerdicts(verdicts, theme) {
  const labels = VERDICTS.map((level) => t(`verdict.${level}`));
  const counts = VERDICTS.map((level) => verdicts[level] ?? 0);
  tableRows(
    $("#verdicts-table"),
    labels.map((label, i) => [label, formatNumber(counts[i])]),
  );
  if (!window.Chart) return;

  drawChart("verdicts", $("#verdicts-chart"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: counts,
          backgroundColor: [theme.safe, theme.warn, theme.danger],
          borderColor: theme.surface,
          borderWidth: 2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      cutout: "60%",
      plugins: { legend: { position: "bottom", labels: { color: theme.text, usePointStyle: true } } },
    },
  });
}

function renderDaily(daily, days, theme) {
  const filled = fillDays(daily, days);
  tableRows(
    $("#daily-table"),
    filled
      .filter((day) => day.total > 0)
      .map((day) => [formatDay(day.date), formatNumber(day.total), formatNumber(day.likely_scam)]),
  );
  if (!window.Chart) return;

  drawChart("daily", $("#daily-chart"), {
    type: "line",
    data: {
      labels: filled.map((day) => formatDay(day.date)),
      datasets: [
        {
          label: t("dash.dailyTotal"),
          data: filled.map((day) => day.total),
          borderColor: theme.brand,
          backgroundColor: theme.brand,
          tension: 0.3,
          pointRadius: days > 30 ? 0 : 2,
        },
        {
          label: t("dash.dailyScams"),
          data: filled.map((day) => day.likely_scam),
          borderColor: theme.danger,
          backgroundColor: theme.danger,
          tension: 0.3,
          pointRadius: days > 30 ? 0 : 2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { color: theme.text, usePointStyle: true } } },
      scales: {
        x: { ticks: { color: theme.text, maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0, color: theme.text }, grid: { color: theme.grid } },
      },
    },
  });
}

function render() {
  const stats = state.stats;
  if (!stats) return;
  const theme = chartTheme();

  if (window.Chart) {
    window.Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    window.Chart.defaults.animation = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? false
      : { duration: 400 };
  } else {
    // Chart.js could not load (offline, blocked CDN): the tables still show the data.
    document.querySelectorAll(".data-table").forEach((details) => {
      details.open = true;
    });
    document.querySelectorAll(".chart").forEach((chart) => {
      chart.hidden = true;
    });
  }

  renderTiles(stats);
  $("#dash-empty").hidden = stats.total > 0;
  renderTypes(stats.scam_types, theme);
  renderVerdicts(stats.verdicts, theme);
  renderDaily(stats.daily, stats.days, theme);
}

// --- Data --------------------------------------------------------------------

async function load() {
  const error = $("#dash-error");
  error.hidden = true;
  try {
    const params = new URLSearchParams({ days: String(state.days), language: state.language });
    const response = await fetch(`/api/stats?${params}`);
    if (!response.ok) throw new Error(String(response.status));
    state.stats = await response.json();
    render();
  } catch {
    error.textContent = t("dash.error");
    error.hidden = false;
  }
}

function applyLanguage(language) {
  state.language = language;
  applyStaticText(language);
  document.title = t("dash.pageTitle");
  document.querySelectorAll("[data-days]").forEach((button) => {
    button.textContent = t("dash.days", { n: button.dataset.days });
  });
}

function init() {
  applyLanguage(initialLanguage());

  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => {
      applyLanguage(button.dataset.lang);
      storeLanguage(button.dataset.lang);
      load(); // scam type labels come from the API in the chosen language
    });
  });

  document.querySelectorAll("[data-days]").forEach((button) => {
    button.addEventListener("click", () => {
      state.days = Number(button.dataset.days);
      document.querySelectorAll("[data-days]").forEach((other) => {
        other.setAttribute("aria-pressed", String(other === button));
      });
      load();
    });
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  load();
}

init();
