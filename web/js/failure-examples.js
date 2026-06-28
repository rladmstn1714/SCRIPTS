const FAILURE_TYPES = {
  terms: {
    label: "Failure to Understand Terms of Address and References",
    short: "Terms of address vs. reference",
  },
  aggregate: {
    label: "Failure to Aggregate Multiple Cues",
    short: "Aggregate multiple cues",
  },
  atypical: {
    label: "Failure to Recognize Beyond Typical Relationships",
    short: "Atypical relationships",
  },
  culture: {
    label: "Failure to Understand Language- or Culture-Specific Features (Ko)",
    short: "Language / culture-specific (Korean)",
  },
};

const state = { active: "terms", initialized: false };

function setActiveFailure(type) {
  if (!FAILURE_TYPES[type]) return;
  state.active = type;
  render();
}

function bindInteractions(root) {
  root.querySelectorAll("[data-failure-type]").forEach((element) => {
    element.addEventListener("click", () => {
      setActiveFailure(element.getAttribute("data-failure-type"));
    });
  });
}

function render() {
  const mount = document.getElementById("failureExplorer");
  if (!mount) return;

  mount.querySelectorAll("[data-failure-type]").forEach((element) => {
    const type = element.getAttribute("data-failure-type");
    const isActive = type === state.active;
    element.classList.toggle("active", isActive);
    if (element.matches(".legend-btn")) {
      element.setAttribute("aria-pressed", isActive ? "true" : "false");
    }
  });

  mount.querySelectorAll(".failure-panel").forEach((panel) => {
    const isActive = panel.dataset.failurePanel === state.active;
    panel.hidden = !isActive;
    panel.classList.toggle("active", isActive);
  });

  if (state.initialized) {
    mount.querySelector(".failure-example-shell")?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }
  state.initialized = true;
}

function initFailureExplorer() {
  const mount = document.getElementById("failureExplorer");
  if (!mount) return;
  bindInteractions(mount);
  render();
}

initFailureExplorer();
