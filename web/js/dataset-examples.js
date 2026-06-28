const EXAMPLES_URL = new URL("../data/examples.json", import.meta.url);
const SLOT_LABELS = ["a", "b", "c", "d"];

const state = {
  lang: "en",
  enExamples: [],
  koExamples: [],
  activeSlot: "a",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function parseDialogue(dialogue) {
  const turns = [];
  const pattern = /(?:^|\n)(\[?([A-Z])\]?:)\s*/g;
  const matches = [...dialogue.matchAll(pattern)];

  if (!matches.length) {
    return [{ speaker: "?", text: dialogue.trim() }];
  }

  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index];
    const speaker = match[2];
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : dialogue.length;
    const text = dialogue.slice(start, end).trim();
    if (text) {
      turns.push({ speaker, text });
    }
  }

  return turns;
}

function speakerClass(speaker) {
  const classes = {
    A: "speaker-a",
    B: "speaker-b",
    C: "speaker-c",
  };
  return classes[speaker] || "speaker-other";
}

function renderTags(labels, className) {
  if (!labels.length) {
    return '<span class="example-tag empty">—</span>';
  }
  return labels
    .map((label) => `<span class="example-tag ${className}">${escapeHtml(label)}</span>`)
    .join("");
}

function renderTranscript(dialogue) {
  return parseDialogue(dialogue)
    .map((turn) => {
      const speakerClassName = speakerClass(turn.speaker);
      return `
        <p class="dialogue-line ${speakerClassName}">
          <span class="dialogue-speaker">${escapeHtml(turn.speaker)}</span>
          <span class="dialogue-text">${escapeHtml(turn.text)}</span>
        </p>
      `;
    })
    .join("");
}

function currentExamples() {
  return state.lang === "ko" ? state.koExamples : state.enExamples;
}

function slotOffset() {
  return SLOT_LABELS.indexOf(state.activeSlot);
}

function currentExample() {
  const examples = currentExamples();
  return examples[slotOffset()] ?? null;
}

function visibleSlots() {
  return SLOT_LABELS.map((slot, offset) => ({
    slot,
    example: currentExamples()[offset] ?? null,
  }));
}

function renderViewer(example) {
  if (!example) {
    return '<p class="example-viewer-empty">No example available for this slot.</p>';
  }

  return `
    <div class="example-viewer">
      <div class="example-label-row">
        <span class="example-label-title">Highly likely</span>
        <span class="example-tags">${renderTags(example.highly_likely, "likely")}</span>
      </div>
      <div class="example-label-row">
        <span class="example-label-title">Unlikely</span>
        <span class="example-tags">${renderTags(example.unlikely, "unlikely")}</span>
      </div>
      <div class="dialogue-transcript">${renderTranscript(example.dialogue)}</div>
    </div>
  `;
}

function renderBoard() {
  const slots = visibleSlots();

  return `
    <div class="examples-board">
      <div class="examples-board-toolbar">
        <div class="examples-lang-toggle" role="tablist" aria-label="Language">
          <button
            class="examples-lang-btn${state.lang === "en" ? " active" : ""}"
            type="button"
            role="tab"
            aria-selected="${state.lang === "en" ? "true" : "false"}"
            data-lang="en"
          >English</button>
          <button
            class="examples-lang-btn${state.lang === "ko" ? " active" : ""}"
            type="button"
            role="tab"
            aria-selected="${state.lang === "ko" ? "true" : "false"}"
            data-lang="ko"
          >Korean</button>
        </div>

        <div class="examples-slot-tabs" role="tablist" aria-label="Example slots">
          ${slots
            .map(
              ({ slot, example }) => `
            <button
              class="examples-slot-tab${slot === state.activeSlot ? " active" : ""}"
              type="button"
              role="tab"
              aria-selected="${slot === state.activeSlot ? "true" : "false"}"
              data-slot-select="${slot}"
              ${example ? "" : "disabled"}
            >${slot}</button>
          `,
            )
            .join("")}
        </div>
      </div>

      <div class="example-viewer-shell" aria-live="polite">
        ${renderViewer(currentExample())}
      </div>
    </div>
  `;
}

function setLanguage(lang) {
  state.lang = lang;
  state.activeSlot = "a";
  render();
}

function setActiveSlot(slot) {
  state.activeSlot = slot;
  render();
}

function bindBoard(root) {
  root.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => {
      const lang = button.getAttribute("data-lang");
      if (lang) setLanguage(lang);
    });
  });

  root.querySelectorAll("[data-slot-select]").forEach((button) => {
    button.addEventListener("click", () => {
      const slot = button.getAttribute("data-slot-select");
      if (slot) setActiveSlot(slot);
    });
  });
}

function render() {
  const mount = document.getElementById("datasetExamples");
  if (!mount) return;

  mount.innerHTML = renderBoard();
  bindBoard(mount);
}

async function initDatasetExamples() {
  const mount = document.getElementById("datasetExamples");
  if (!mount) return;

  try {
    const response = await fetch(EXAMPLES_URL);
    if (!response.ok) throw new Error("Failed to load examples");
    const payload = await response.json();

    state.enExamples = (payload.examples?.en ?? []).slice(0, 4);
    state.koExamples = (payload.examples?.ko ?? []).slice(0, 4);
    state.lang = "en";
    state.activeSlot = "a";
    render();
  } catch (error) {
    console.error(error);
    mount.innerHTML = '<p class="example-error">Could not load dataset examples.</p>';
  }
}

initDatasetExamples();
