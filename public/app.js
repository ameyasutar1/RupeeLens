const state = { data: null, visibleTransactions: 40 };
let toastTimer;
let forecastTimer;
let agentThreadId = localStorage.getItem("arthaThreadId") || crypto.randomUUID();
localStorage.setItem("arthaThreadId", agentThreadId);

async function loadAccount() {
  const response = await fetch("/api/auth/me");
  if (response.status === 401) {
    window.location.replace("/login.html");
    return false;
  }
  const payload = await response.json();
  document.querySelector("#accountName").textContent = payload.user.display_name;
  document.querySelector("#userGreeting").textContent =
    `${payload.user.display_name.toUpperCase()}'S FINANCIAL FIELD NOTES`;
  return true;
}

async function logout() {
  await fetch("/api/auth/logout", { method: "POST" });
  localStorage.removeItem("arthaThreadId");
  window.location.replace("/login.html");
}

const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
}[character]));

function renderMarkdown(value) {
  const inline = text => escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

  const lines = String(value).replace(/\r\n/g, "\n").split("\n");
  const output = [];
  let listType = null;

  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const nextLine = lines[index + 1] || "";
    const isTableHeader = line.includes("|") &&
      /^\s*\|?[\s:-]+(?:\|[\s:-]+)+\|?\s*$/.test(nextLine);

    if (isTableHeader) {
      closeList();
      const splitRow = row => row.trim().replace(/^\||\|$/g, "")
        .split("|").map(cell => cell.trim());
      const headers = splitRow(line);
      const rows = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(splitRow(lines[index]));
        index += 1;
      }
      index -= 1;
      output.push(`<div class="markdown-table-wrap"><table><thead><tr>${
        headers.map(cell => `<th>${inline(cell)}</th>`).join("")
      }</tr></thead><tbody>${
        rows.map(row => `<tr>${row.map(cell => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("")
      }</tbody></table></div>`);
      continue;
    }

    if (unordered || ordered) {
      const nextType = unordered ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        output.push(`<${nextType}>`);
        listType = nextType;
      }
      output.push(`<li>${inline((unordered || ordered)[1])}</li>`);
      continue;
    }

    closeList();
    if (!line.trim()) {
      output.push("");
    } else if (heading) {
      const level = heading[1].length + 2;
      output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else {
      output.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return output.join("");
}

const money = (value, compact = false) => new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: compact ? 0 : 2,
  notation: compact && Math.abs(value) >= 100000 ? "compact" : "standard",
}).format(value);

const shortDate = value => new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", {
  day: "2-digit", month: "short", year: "numeric",
});

const monthName = value => new Date(`${value}-01T00:00:00`).toLocaleDateString("en-IN", {
  month: "short", year: "2-digit",
});

async function loadDashboard() {
  const params = new URLSearchParams();
  const start = document.querySelector("#startDate").value;
  const end = document.querySelector("#endDate").value;
  const category = document.querySelector("#categoryFilter").value;
  const direction = document.querySelector("#directionFilter").value;
  const channel = document.querySelector("#channelFilter").value;
  const minimum = document.querySelector("#minAmount").value;
  const maximum = document.querySelector("#maxAmount").value;
  const query = document.querySelector("#searchInput").value.trim();
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  if (category !== "all") params.set("category", category);
  if (direction !== "all") params.set("direction", direction);
  if (channel !== "all") params.set("channel", channel);
  if (minimum) params.set("min", minimum);
  if (maximum) params.set("max", maximum);
  if (query) params.set("q", query);

  document.body.classList.add("loading");
  try {
    const response = await fetch(`/api/dashboard?${params}`);
    state.data = await response.json();
    state.visibleTransactions = 40;
    render();
    populateFilters();
    document.querySelector("#databaseCount").textContent =
      `${state.data.summary.transactionCount.toLocaleString("en-IN")} records in the current view`;
  } catch (error) {
    showToast("Could not load the dashboard. Is the Python server running?", true);
  } finally {
    document.body.classList.remove("loading");
  }
}

function render() {
  const { period, summary } = state.data;
  document.querySelector("#periodLabel").textContent = period.start
    ? `${shortDate(period.start)} — ${shortDate(period.end)}`
    : "No matching activity";
  document.querySelector("#transactionLabel").textContent = `${summary.transactionCount.toLocaleString("en-IN")} transactions analyzed`;
  document.querySelector("#spendTotal").textContent = money(summary.spend, true);
  document.querySelector("#incomeTotal").textContent = money(summary.income, true);
  document.querySelector("#transferTotal").textContent = money(summary.transfers, true);
  document.querySelector("#investmentTotal").textContent = money(summary.investments, true);
  document.querySelector("#monthlyAverage").textContent = money(summary.averageMonthly, true);
  document.querySelector("#netTotal").textContent = money(summary.net, true);
  document.querySelector("#donutTotal").textContent = money(summary.spend, true);
  renderCategories();
  renderInsights();
  renderMonthlyChart();
  renderLists();
  renderTransactions();
}

function renderCategories() {
  const categories = state.data.categories;
  const categoryList = document.querySelector("#categoryList");
  categoryList.innerHTML = categories.map(item => `
    <div class="category-row" data-category="${escapeHtml(item.name)}">
      <i style="background:${item.color}"></i>
      <div>${escapeHtml(item.name)}<small>${item.count} payments · ${item.share}%</small></div>
      <strong>${money(item.amount, true)}</strong>
    </div>
  `).join("") || `<p class="empty">No expenses in this view.</p>`;
  categoryList.querySelectorAll(".category-row").forEach(row => {
    row.addEventListener("click", () => {
      document.querySelector("#categoryFilter").value = row.dataset.category;
      loadDashboard();
    });
  });

  let cursor = 0;
  const segments = categories.map(item => {
    const start = cursor;
    cursor += item.share;
    return `${item.color} ${start}% ${cursor}%`;
  });
  document.querySelector("#categoryDonut").style.background =
    segments.length ? `conic-gradient(${segments.join(",")})` : "#dedbd2";
}

function renderInsights() {
  document.querySelector("#insightList").innerHTML = state.data.insights.map(item => `
    <article class="insight"><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.text)}</p></article>
  `).join("") || `<p class="empty">Adjust the filters to reveal insights.</p>`;
}

function renderMonthlyChart() {
  const svg = document.querySelector("#monthlyChart");
  const data = state.data.monthly;
  if (!data.length) {
    svg.innerHTML = "";
    return;
  }
  const width = Math.max(900, data.length * 70);
  const height = 300;
  const pad = { top: 25, right: 20, bottom: 45, left: 62 };
  const plotHeight = height - pad.top - pad.bottom;
  const max = Math.max(...data.flatMap(item => [item.spend, item.income]), 1);
  const y = value => pad.top + plotHeight - (value / max) * plotHeight;
  const x = index => pad.left + index * ((width - pad.left - pad.right) / Math.max(data.length - 1, 1));
  const line = key => data.map((item, index) => `${index ? "L" : "M"} ${x(index)} ${y(item[key])}`).join(" ");

  const grids = [0, .25, .5, .75, 1].map(ratio => {
    const ypos = pad.top + plotHeight * ratio;
    const label = money(max * (1 - ratio), true);
    return `<line x1="${pad.left}" x2="${width-pad.right}" y1="${ypos}" y2="${ypos}" stroke="rgba(23,32,27,.12)" />
      <text x="${pad.left-10}" y="${ypos+3}" text-anchor="end" font-size="9" fill="#6f756e">${label}</text>`;
  }).join("");
  const labels = data.map((item, index) =>
    `<text x="${x(index)}" y="${height-16}" text-anchor="middle" font-size="9" fill="#6f756e">${monthName(item.month)}</text>`
  ).join("");
  const dots = data.map((item, index) => `
    <circle cx="${x(index)}" cy="${y(item.spend)}" r="3.5" fill="#ff6b45"><title>${monthName(item.month)} spend: ${money(item.spend)}</title></circle>
    <circle cx="${x(index)}" cy="${y(item.income)}" r="3.5" fill="#173f35"><title>${monthName(item.month)} income: ${money(item.income)}</title></circle>
  `).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grids}${labels}
    <path d="${line("spend")}" fill="none" stroke="#ff6b45" stroke-width="2.5" />
    <path d="${line("income")}" fill="none" stroke="#173f35" stroke-width="2.5" />
    ${dots}`;
}

function listMarkup(items, recurring = false) {
  return items.map((item, index) => `
    <div class="merchant-row">
      <span class="merchant-index">${String(index + 1).padStart(2, "0")}</span>
      <div>
        <strong>${escapeHtml(item.merchant)}</strong>
        <small>${recurring
          ? `${item.count} payments across ${item.months} months · avg ${money(item.average)}`
          : `${shortDate(item.date)} · ${item.category}`}</small>
      </div>
      <span class="merchant-amount">${money(recurring ? item.total : item.amount, true)}</span>
    </div>
  `).join("") || `<p class="empty">Nothing to show in this view.</p>`;
}

function renderLists() {
  document.querySelector("#recurringList").innerHTML = listMarkup(state.data.recurring, true);
  document.querySelector("#largestList").innerHTML = listMarkup(state.data.largest);
}

function renderTransactions() {
  const transactions = state.data.transactions;
  const visible = transactions.slice(0, state.visibleTransactions);
  document.querySelector("#ledgerCount").textContent = `${transactions.length.toLocaleString("en-IN")} records`;
  document.querySelector("#transactionTable").innerHTML = visible.map(item => `
    <tr>
      <td>${shortDate(item.date)}</td>
      <td class="transaction-detail"><strong>${escapeHtml(item.merchant)}</strong><small title="${escapeHtml(item.description)}">${escapeHtml(item.description)}</small></td>
      <td><span class="category-pill">${escapeHtml(item.category)}</span></td>
      <td>${escapeHtml(item.channel)}</td>
      <td class="${item.direction === "income" ? "income-amount" : "expense-amount"}">
        ${item.direction === "income" ? "+" : "−"}${money(item.amount)}
      </td>
    </tr>
  `).join("");
  document.querySelector("#loadMore").hidden = state.visibleTransactions >= transactions.length;
}

function populateFilters() {
  const categorySelect = document.querySelector("#categoryFilter");
  const selectedCategory = categorySelect.value;
  categorySelect.innerHTML = `<option value="all">All activity</option>` +
    state.data.configuration.categories.map(item =>
      `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`
    ).join("");
  categorySelect.value = [...categorySelect.options].some(option => option.value === selectedCategory)
    ? selectedCategory : "all";

  const channelSelect = document.querySelector("#channelFilter");
  const selectedChannel = channelSelect.value;
  channelSelect.innerHTML = `<option value="all">All methods</option>` +
    state.data.configuration.channels.map(item =>
      `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`
    ).join("");
  channelSelect.value = [...channelSelect.options].some(option => option.value === selectedChannel)
    ? selectedChannel : "all";
}

function showToast(message, error = false) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 5200);
}

async function uploadStatements(files) {
  if (!files.length) return;
  const form = new FormData();
  [...files].forEach(file => form.append("statements", file));
  document.querySelector("#databaseCount").textContent = `Importing ${files.length} file${files.length > 1 ? "s" : ""}…`;
  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Import failed");
    const added = payload.results.reduce((total, result) => total + result.added, 0);
    const skipped = payload.results.reduce((total, result) => total + result.skipped, 0);
    showToast(`${added} new transactions added · ${skipped} overlapping rows skipped`);
    renderImportHistory(payload.history);
    await loadDashboard();
    if (added) await loadIntelligence();
    if (added) await loadForecast(Number(document.querySelector("#forecastHorizon").value));
  } catch (error) {
    showToast(error.message, true);
    document.querySelector("#databaseCount").textContent = "Import failed — your existing data is unchanged";
  } finally {
    document.querySelector("#statementFiles").value = "";
  }
}

async function loadImportHistory() {
  try {
    const response = await fetch("/api/imports");
    renderImportHistory(await response.json());
  } catch {
    showToast("Could not read import history.", true);
  }
}

function renderImportHistory(items) {
  const panel = document.querySelector("#importHistory");
  panel.innerHTML = items.map(item => `
    <div class="history-row">
      <strong>${escapeHtml(item.filename)}</strong>
      <span>${item.rows_added} added · ${item.rows_skipped} skipped</span>
      <span>${new Date(item.imported_at).toLocaleString("en-IN")}</span>
    </div>
  `).join("") || `<div class="history-row">No statements imported yet.</div>`;
}

function resetFilters() {
  ["startDate", "endDate", "minAmount", "maxAmount", "searchInput"].forEach(id => {
    document.querySelector(`#${id}`).value = "";
  });
  ["categoryFilter", "directionFilter", "channelFilter"].forEach(id => {
    document.querySelector(`#${id}`).value = "all";
  });
  loadDashboard();
}

function addChatMessage(role, content, temporary = false) {
  const container = document.querySelector("#chatMessages");
  const message = document.createElement("div");
  message.className = `message ${role}${temporary ? " thinking" : ""}`;
  if (temporary) message.id = "agentThinking";
  const label = document.createElement("span");
  label.textContent = role === "user" ? "You" : "Artha";
  const paragraph = document.createElement(
    role === "assistant" && !temporary ? "div" : "p"
  );
  if (role === "assistant" && !temporary) {
    paragraph.classList.add("markdown-content");
    paragraph.innerHTML = renderMarkdown(content);
  } else {
    paragraph.textContent = content;
  }
  message.append(label, paragraph);
  container.append(message);
  container.scrollTop = container.scrollHeight;
}

async function sendAgentMessage(message) {
  addChatMessage("user", message);
  addChatMessage("assistant", "Reviewing your ledger", true);
  document.querySelector("#chatInput").disabled = true;
  try {
    const response = await fetch("/api/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: agentThreadId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Agent request failed");
    document.querySelector("#agentThinking")?.remove();
    addChatMessage("assistant", payload.message);
    renderPendingActions(payload.actions, payload.rule_actions);
  } catch (error) {
    document.querySelector("#agentThinking")?.remove();
    addChatMessage("assistant", `I hit a problem: ${error.message}`);
  } finally {
    document.querySelector("#chatInput").disabled = false;
    document.querySelector("#chatInput").focus();
  }
}

async function loadAgentSession() {
  try {
    const [messagesResponse, actionsResponse, rulesResponse] = await Promise.all([
      fetch(`/api/agent/messages?thread_id=${encodeURIComponent(agentThreadId)}`),
      fetch("/api/agent/actions"),
      fetch("/api/agent/rules"),
    ]);
    const messages = await messagesResponse.json();
    const actions = await actionsResponse.json();
    const rules = await rulesResponse.json();
    if (messages.length) {
      document.querySelector("#chatMessages").innerHTML = "";
      messages.forEach(item => addChatMessage(item.role, item.content));
    }
    renderPendingActions(actions, rules);
  } catch {
    // The dashboard remains fully usable without the agent panel.
  }
}

async function loadIntelligence(force = false) {
  const button = document.querySelector("#refreshIntelligence");
  button.disabled = true;
  button.textContent = "Analyzing…";
  try {
    const response = await fetch(`/api/intelligence${force ? "?force=true" : ""}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Intelligence pipeline failed");
    renderIntelligence(payload);
  } catch (error) {
    document.querySelector("#intelligenceHeadline").textContent = "The adaptive narrative is temporarily unavailable.";
    document.querySelector("#intelligenceSummary").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Re-run analysis";
  }
}

function renderIntelligence(payload) {
  const narrative = payload.narrative;
  document.querySelector("#intelligenceHeadline").textContent = narrative.headline;
  document.querySelector("#intelligenceSummary").textContent = narrative.executive_summary;
  document.querySelector("#adaptiveCards").innerHTML = narrative.cards.map(card => `
    <article data-tone="${escapeHtml(card.tone)}">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <small>${escapeHtml(card.context)}</small>
    </article>
  `).join("");
  document.querySelector("#adaptiveFindings").innerHTML = narrative.findings
    .map(item => `<div class="adaptive-item">${escapeHtml(item)}</div>`).join("");
  document.querySelector("#adaptiveWatchlist").innerHTML = (narrative.watchlist.length
    ? narrative.watchlist : ["No material anomaly is currently on the watchlist."])
    .map(item => `<div class="adaptive-item">${escapeHtml(item)}</div>`).join("");
  document.querySelector("#pipelineMeta").textContent =
    `${payload.source} narrative · data version ${payload.version} · ${new Date(payload.created_at).toLocaleString("en-IN")}`;

  if (narrative.recommended_questions?.length) {
    const starterBox = document.querySelector(".prompt-starters");
    starterBox.innerHTML = narrative.recommended_questions.map(question =>
      `<button>${escapeHtml(question)}</button>`
    ).join("");
    bindPromptStarters();
  }
}

async function loadForecast(horizon = 7) {
  document.querySelector("#forecastNarrative").textContent =
    "Running the latest model over your spending patterns…";
  try {
    const response = await fetch(`/api/forecast?horizon=${horizon}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Forecast failed");
    renderForecast(payload);
  } catch (error) {
    document.querySelector("#forecastNarrative").textContent = error.message;
    document.querySelector("#forecastConfidence").textContent = "Unavailable";
  }
}

function renderForecast(payload) {
  const top = payload.top_category;
  const metrics = payload.model.metrics;
  const confidence = metrics.r2 > .35
    ? "Moderate" : metrics.r2 > 0 ? "Experimental" : "Low";
  document.querySelector("#forecastDaysTitle").textContent = payload.horizon_days;
  document.querySelector("#forecastTotal").textContent = money(payload.predicted_total, true);
  document.querySelector("#forecastAverage").textContent = money(payload.predicted_daily_average, true);
  document.querySelector("#forecastTopCategory").textContent = top?.category || "No signal";
  document.querySelector("#forecastTopShare").textContent = top
    ? `${top.share}% · ${money(top.amount, true)}` : "No category forecast";
  document.querySelector("#forecastRange").textContent =
    `${money(payload.uncertainty.lower_total, true)}–${money(payload.uncertainty.upper_total, true)} estimated range`;
  document.querySelector("#forecastConfidence").textContent = confidence;
  document.querySelector("#forecastMetric").textContent =
    `R² ${metrics.r2} · MAE ${money(metrics.mae, true)}`;
  document.querySelector("#forecastNarrative").textContent = top
    ? `The model expects ${top.category} to lead regular spending from ${shortDate(payload.forecast_start)} to ${shortDate(payload.forecast_end)}. This excludes rare categories and statistical outliers.`
    : "The model found insufficient repeatable spending patterns.";
  document.querySelector("#forecastTrainingNote").textContent =
    `${payload.methodology.algorithm} · ${payload.model.train_rows.toLocaleString("en-IN")} train / ${payload.model.test_rows.toLocaleString("en-IN")} test rows`;
  document.querySelector("#forecastExclusions").textContent =
    `${payload.model.outliers_removed} outliers removed · excluded sparse: ${payload.model.excluded_categories.join(", ") || "none"}`;
  document.querySelector("#forecastRetrained").textContent =
    `trained ${new Date(payload.model.trained_at).toLocaleString("en-IN")} · auto-retrains after imports`;

  document.querySelector("#forecastCategories").innerHTML = payload.category_forecast
    .slice(0, 7).map(item => `
      <div class="forecast-category-row">
        <header><span>${escapeHtml(item.category)}</span><span>${money(item.amount, true)} · ${item.share}%</span></header>
        <div class="forecast-bar"><i style="width:${item.share}%;background:${item.color}"></i></div>
      </div>`).join("");
  renderForecastChart(payload.daily_forecast);
}

function renderForecastChart(data) {
  const svg = document.querySelector("#forecastChart");
  const chartWrap = document.querySelector(".forecast-chart-wrap");
  let tooltip = chartWrap.querySelector(".forecast-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = "forecast-tooltip";
    chartWrap.append(tooltip);
  }
  const width = Math.max(700, data.length * 44);
  const height = 280;
  const pad = { top: 24, right: 18, bottom: 45, left: 58 };
  const max = Math.max(...data.map(item => item.total), 1);
  const plotHeight = height - pad.top - pad.bottom;
  const slot = (width - pad.left - pad.right) / data.length;
  const barWidth = Math.max(9, slot * .55);
  const grids = [0, .5, 1].map(ratio => {
    const y = pad.top + plotHeight * ratio;
    return `<line x1="${pad.left}" x2="${width-pad.right}" y1="${y}" y2="${y}" stroke="rgba(255,255,255,.12)"/>
      <text x="${pad.left-9}" y="${y+3}" text-anchor="end" fill="rgba(255,255,255,.5)" font-size="8">${money(max*(1-ratio), true)}</text>`;
  }).join("");
  const bars = data.map((item, index) => {
    const x = pad.left + slot * index + (slot - barWidth) / 2;
    const barHeight = item.total / max * plotHeight;
    const y = pad.top + plotHeight - barHeight;
    const label = new Date(`${item.date}T00:00:00`).toLocaleDateString(
      "en-IN", { day: "2-digit", month: "short" }
    );
    return `<rect class="forecast-bar-visual" data-forecast-index="${index}" x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="#c8ee75" rx="2"></rect>
      <rect class="forecast-hit-area" data-forecast-index="${index}" x="${pad.left + slot * index}" y="${pad.top}" width="${slot}" height="${plotHeight}" fill="transparent"></rect>
      <text x="${x+barWidth/2}" y="${height-17}" text-anchor="middle" fill="rgba(255,255,255,.5)" font-size="8">${label}</text>`;
  }).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = grids + bars;

  const hideTooltip = () => {
    tooltip.classList.remove("visible");
    svg.querySelectorAll(".forecast-bar-visual").forEach(bar => bar.classList.remove("hovered"));
  };
  svg.querySelectorAll(".forecast-hit-area").forEach(hitArea => {
    hitArea.addEventListener("mouseenter", event => {
      const index = Number(event.currentTarget.dataset.forecastIndex);
      const item = data[index];
      const topCategories = item.categories
        .filter(category => category.amount > 0)
        .slice(0, 5);
      tooltip.innerHTML = `
        <strong>${shortDate(item.date)}</strong>
        <span>Predicted total · ${money(item.total)}</span>
        ${topCategories.map(category => {
          const categoryMeta = state.data?.configuration?.categories
            ?.find(meta => meta.name === category.category);
          return `<div class="forecast-tooltip-row">
            <i style="background:${categoryMeta?.color || "#94a0ad"}"></i>
            <span>${escapeHtml(category.category)}</span>
            <b>${money(category.amount)}</b>
          </div>`;
        }).join("")}
      `;
      const svgRect = svg.getBoundingClientRect();
      const wrapRect = chartWrap.getBoundingClientRect();
      const pointerX = event.clientX - wrapRect.left + chartWrap.scrollLeft;
      const pointerY = event.clientY - wrapRect.top;
      const clampedX = Math.max(125, Math.min(pointerX, chartWrap.scrollWidth - 125));
      tooltip.style.left = `${clampedX}px`;
      tooltip.style.top = `${Math.max(150, pointerY)}px`;
      tooltip.classList.add("visible");
      svg.querySelectorAll(".forecast-bar-visual").forEach(bar => bar.classList.remove("hovered"));
      svg.querySelector(`.forecast-bar-visual[data-forecast-index="${index}"]`)?.classList.add("hovered");
    });
    hitArea.addEventListener("mousemove", event => {
      const wrapRect = chartWrap.getBoundingClientRect();
      const pointerX = event.clientX - wrapRect.left + chartWrap.scrollLeft;
      tooltip.style.left = `${Math.max(125, Math.min(pointerX, chartWrap.scrollWidth - 125))}px`;
    });
    hitArea.addEventListener("mouseleave", hideTooltip);
  });
  chartWrap.addEventListener("scroll", hideTooltip, { passive: true });
}

function bindPromptStarters() {
  document.querySelectorAll(".prompt-starters button").forEach(button => {
    button.onclick = () => {
      document.querySelector("#chatInput").value = button.textContent;
      document.querySelector("#chatForm").requestSubmit();
    };
  });
}

function renderPendingActions(actions = [], rules = []) {
  actions = actions || [];
  rules = rules || [];
  const container = document.querySelector("#pendingActions");
  const transactionCards = actions.map(action => {
    const changes = Object.entries(action.proposed_changes)
      .map(([field, value]) => field === "_remember_for_future"
        ? "remember this merchant for future imports"
        : `${escapeHtml(field)} → ${escapeHtml(value)}`)
      .join(" · ");
    return `
      <article class="action-card">
        <div>
          <span>Approval required · action ${action.id}</span>
          <strong>${escapeHtml(action.merchant)} · ${money(action.amount)} · ${shortDate(action.transaction_date)}</strong>
          <p>${escapeHtml(action.reason)}</p>
          <div class="action-changes">${changes}</div>
        </div>
        <div class="action-buttons">
          <button data-action="${action.id}" data-decision="reject">Reject</button>
          <button class="approve" data-action="${action.id}" data-decision="approve">Approve edit</button>
        </div>
      </article>`;
  });
  const ruleCards = rules.map(rule => {
    const samples = rule.sample_transactions.slice(0, 3)
      .map(item => `${escapeHtml(item.merchant)} · ${money(item.amount)} · ${shortDate(item.transaction_date)}`)
      .join("<br>");
    return `
      <article class="action-card rule-action-card">
        <div>
          <span>Rule learning approval · proposal ${rule.id}</span>
          <strong>${escapeHtml(rule.pattern)} → ${escapeHtml(rule.category)}</strong>
          <p>${escapeHtml(rule.reason)}</p>
          <div class="action-changes">
            Match ${escapeHtml(rule.match_field)} · ${rule.affected_count} existing Other transaction${rule.affected_count === 1 ? "" : "s"}
            ${rule.apply_to_existing ? " will be reclassified" : " will remain unchanged"}
          </div>
          <div class="rule-samples">${samples}</div>
        </div>
        <div class="action-buttons">
          <button data-rule="${rule.id}" data-decision="reject">Reject</button>
          <button class="approve" data-rule="${rule.id}" data-decision="approve">Teach Artha</button>
        </div>
      </article>`;
  });
  container.innerHTML = [...ruleCards, ...transactionCards].join("");
  container.querySelectorAll("button[data-action]").forEach(button => {
    button.addEventListener("click", () => resolveAgentAction(button.dataset.action, button.dataset.decision));
  });
  container.querySelectorAll("button[data-rule]").forEach(button => {
    button.addEventListener("click", () => resolveRuleProposal(button.dataset.rule, button.dataset.decision));
  });
}

async function resolveAgentAction(actionId, decision) {
  try {
    const response = await fetch(`/api/agent/actions/${actionId}/${decision}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    showToast(decision === "approve" ? "Ledger correction approved and audited." : "Proposed change rejected.");
    await refreshPendingApprovals();
    if (decision === "approve") await loadDashboard();
    if (decision === "approve") await loadIntelligence();
  } catch (error) {
    showToast(error.message || "Could not resolve the action.", true);
  }
}

async function refreshPendingApprovals() {
  const [actionsResponse, rulesResponse] = await Promise.all([
    fetch("/api/agent/actions"),
    fetch("/api/agent/rules"),
  ]);
  renderPendingActions(await actionsResponse.json(), await rulesResponse.json());
}

async function resolveRuleProposal(proposalId, decision) {
  try {
    const response = await fetch(`/api/agent/rules/${proposalId}/${decision}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    showToast(
      decision === "approve"
        ? `Artha learned the rule and reclassified ${payload.rows_reclassified} transaction${payload.rows_reclassified === 1 ? "" : "s"}.`
        : "Classification rule proposal rejected."
    );
    await refreshPendingApprovals();
    if (decision === "approve") {
      await loadDashboard();
      await loadIntelligence(true);
      await loadForecast();
    }
  } catch (error) {
    showToast(error.message || "Could not resolve the rule proposal.", true);
  }
}

document.querySelector("#applyFilters").addEventListener("click", loadDashboard);
document.querySelector("#resetFilters").addEventListener("click", resetFilters);
document.querySelector("#refreshButton").addEventListener("click", loadDashboard);
document.querySelector("#logoutButton").addEventListener("click", logout);
document.querySelector("#openUpload").addEventListener("click", () => {
  document.querySelector("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => document.querySelector("#statementFiles").click(), 450);
});
document.querySelector("#statementFiles").addEventListener("change", event => uploadStatements(event.target.files));
document.querySelector("#showHistory").addEventListener("click", async () => {
  const panel = document.querySelector("#importHistory");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) await loadImportHistory();
});
const dropZone = document.querySelector("#dropZone");
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", event => uploadStatements(event.dataTransfer.files));
document.querySelector("#searchInput").addEventListener("keydown", event => {
  if (event.key === "Enter") loadDashboard();
});
document.querySelector("#loadMore").addEventListener("click", () => {
  state.visibleTransactions += 50;
  renderTransactions();
});
document.querySelector("#chatForm").addEventListener("submit", event => {
  event.preventDefault();
  const input = document.querySelector("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  sendAgentMessage(message);
});
document.querySelector("#chatInput").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.querySelector("#chatForm").requestSubmit();
  }
});
bindPromptStarters();
document.querySelector("#refreshIntelligence").addEventListener("click", () => loadIntelligence(true));
document.querySelector("#forecastHorizon").addEventListener("input", event => {
  const horizon = Number(event.target.value);
  document.querySelector("#forecastHorizonValue").textContent = `${horizon} days`;
  document.querySelector("#forecastDaysTitle").textContent = horizon;
  clearTimeout(forecastTimer);
  forecastTimer = setTimeout(() => loadForecast(horizon), 300);
});
document.querySelector("#clearChat").addEventListener("click", () => {
  agentThreadId = crypto.randomUUID();
  localStorage.setItem("arthaThreadId", agentThreadId);
  document.querySelector("#chatMessages").innerHTML = "";
  addChatMessage("assistant", "Fresh page. What would you like to understand?");
});

async function startDashboard() {
  if (!await loadAccount()) return;
  await Promise.all([
    loadDashboard(),
    loadAgentSession(),
    loadIntelligence(),
    loadForecast(7),
  ]);
}

startDashboard();
