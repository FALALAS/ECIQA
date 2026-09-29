const {
  COLOR_FIX_COVERAGE_MODE,
  CSV_HEADER,
  buildScenes,
  buildTrials,
  canonicalPairId,
  canonicalSeed,
  choicesForTrial,
  imageOrientation,
  resultToCsvRow,
  summarizeRanking,
} = window.MosCore;
const {
  GROUP_IDS,
  comparisonMarksToTrials,
  createGroupCsvRepository,
  isComparisonMarked,
  selectUnfinishedTrials,
} = window.MosGroupCsv;

const records = window.MOS_RECORDS || [];
const meta = window.MOS_MANIFEST_META || {};
const scenes = buildScenes(records);

const els = {
  metaText: document.getElementById("metaText"),
  expertId: document.getElementById("expertId"),
  groupId: document.getElementById("groupId"),
  workingFolderBtn: document.getElementById("workingFolderBtn"),
  workingFolderStatus: document.getElementById("workingFolderStatus"),
  datasetFilter: document.getElementById("datasetFilter"),
  pairMode: document.getElementById("pairMode"),
  maxTrials: document.getElementById("maxTrials"),
  seedText: document.getElementById("seedText"),
  startBtn: document.getElementById("startBtn"),
  progressText: document.getElementById("progressText"),
  sceneText: document.getElementById("sceneText"),
  comparisonMarkBtn: document.getElementById("comparisonMarkBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  exportCsvBtn: document.getElementById("exportCsvBtn"),
  exportJsonBtn: document.getElementById("exportJsonBtn"),
  clearBtn: document.getElementById("clearBtn"),
  comparisonWorkspace: document.getElementById("comparisonWorkspace"),
  imageGrid: document.getElementById("comparisonWorkspace"),
  inputSlot: document.querySelector(".inputPanel"),
  aSlot: document.querySelector(".candidateAPanel"),
  bSlot: document.querySelector(".candidateBPanel"),
  inputFrame: document.querySelector('[data-label="Input"]'),
  aFrame: document.querySelector('[data-label="A"]'),
  bFrame: document.querySelector('[data-label="B"]'),
  inputImage: document.getElementById("inputImage"),
  aImage: document.getElementById("aImage"),
  bImage: document.getElementById("bImage"),
  rankingText: document.getElementById("rankingText"),
};

const state = {
  trials: [],
  index: 0,
  choices: {},
  results: new Map(),
  groupResults: new Map(),
  comparisonMarks: new Map(),
  groupRepository: null,
  directoryHandle: null,
  rememberedDirectoryHandle: null,
  expertId: "anonymous",
  groupId: "A",
  storageKey: "",
  legacyStorageKey: "",
  trialStartedAt: 0,
  writePending: false,
  markWritePending: false,
  emptyMessage: "No unfinished trials",
};

const HANDLE_DB_NAME = "eciqa-pref-working-folder";
const HANDLE_STORE_NAME = "handles";
const HANDLE_KEY = "working-folder";

function safeExpert() {
  return (els.expertId.value || "anonymous").trim() || "anonymous";
}

function safeGroup() {
  const groupId = String(els.groupId.value || "A").trim().toUpperCase();
  return ["A", "B", "C", "D", "E"].includes(groupId) ? groupId : "A";
}

function safeSeed() {
  return canonicalSeed(els.seedText.value || safeExpert() || "sec-mos");
}

function legacyStorageKey(expertId = safeExpert()) {
  return [
    "eciqa-pref",
    expertId,
    els.datasetFilter.value,
    els.pairMode.value,
    els.maxTrials.value,
    safeSeed(),
  ].join(":");
}

function storageKey(expertId = safeExpert(), groupId = safeGroup()) {
  return [
    "eciqa-pref",
    expertId,
    groupId,
    els.datasetFilter.value,
    els.pairMode.value,
    els.maxTrials.value,
    safeSeed(),
  ].join(":");
}

function setMetaText() {
  if (!records.length) {
    els.metaText.textContent = "manifest.js is missing or empty";
    return;
  }
  const sceneCount = meta.scene_count || scenes.length;
  els.metaText.textContent = `${records.length} images, ${sceneCount} scenes`;
}

function setWorkingFolderStatus(text, status = "idle") {
  els.workingFolderStatus.textContent = text;
  els.workingFolderStatus.dataset.state = status;
}

function openHandleDatabase() {
  if (!window.indexedDB) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(HANDLE_DB_NAME, 1);
    request.addEventListener("upgradeneeded", () => {
      if (!request.result.objectStoreNames.contains(HANDLE_STORE_NAME)) {
        request.result.createObjectStore(HANDLE_STORE_NAME);
      }
    });
    request.addEventListener("success", () => resolve(request.result));
    request.addEventListener("error", () => reject(request.error));
  });
}

async function rememberDirectoryHandle(handle) {
  const database = await openHandleDatabase();
  if (!database) return;
  await new Promise((resolve, reject) => {
    const transaction = database.transaction(HANDLE_STORE_NAME, "readwrite");
    transaction.objectStore(HANDLE_STORE_NAME).put(handle, HANDLE_KEY);
    transaction.addEventListener("complete", resolve);
    transaction.addEventListener("error", () => reject(transaction.error));
    transaction.addEventListener("abort", () => reject(transaction.error));
  });
  database.close();
}

async function loadRememberedDirectoryHandle() {
  const database = await openHandleDatabase();
  if (!database) return null;
  const handle = await new Promise((resolve, reject) => {
    const transaction = database.transaction(HANDLE_STORE_NAME, "readonly");
    const request = transaction.objectStore(HANDLE_STORE_NAME).get(HANDLE_KEY);
    request.addEventListener("success", () => resolve(request.result || null));
    request.addEventListener("error", () => reject(request.error));
  });
  database.close();
  return handle;
}

async function directoryPermission(handle, requestPermission = false) {
  if (!handle) return false;
  if (typeof handle.queryPermission !== "function") return true;
  if (await handle.queryPermission({ mode: "readwrite" }) === "granted") return true;
  if (!requestPermission || typeof handle.requestPermission !== "function") return false;
  return await handle.requestPermission({ mode: "readwrite" }) === "granted";
}

async function connectWorkingFolder(handle) {
  const repository = createGroupCsvRepository(handle);
  await repository.initialize();
  state.directoryHandle = handle;
  state.groupRepository = repository;
  state.rememberedDirectoryHandle = handle;
  els.workingFolderBtn.textContent = "Change Folder";
  els.startBtn.disabled = false;
  setWorkingFolderStatus(`Folder: ${handle.name || "selected"}`, "connected");
}

async function chooseWorkingFolder() {
  els.workingFolderBtn.disabled = true;
  try {
    let handle = state.directoryHandle ? null : state.rememberedDirectoryHandle;
    if (handle && !await directoryPermission(handle, true)) handle = null;
    if (!handle) {
      if (typeof window.showDirectoryPicker !== "function") {
        throw new Error("Folder access requires Chrome or Edge");
      }
      handle = await window.showDirectoryPicker({
        id: "eciqa-pref-results",
        mode: "readwrite",
      });
    }
    await connectWorkingFolder(handle);
    try {
      await rememberDirectoryHandle(handle);
    } catch {
      // A selected folder remains usable even if its handle cannot be remembered.
    }
  } catch (error) {
    if (error?.name !== "AbortError") {
      setWorkingFolderStatus(`Folder error: ${error.message}`, "error");
    }
  } finally {
    els.workingFolderBtn.disabled = false;
  }
}

async function restoreWorkingFolder() {
  try {
    const handle = await loadRememberedDirectoryHandle();
    if (!handle) return;
    state.rememberedDirectoryHandle = handle;
    if (await directoryPermission(handle)) {
      await connectWorkingFolder(handle);
    } else {
      setWorkingFolderStatus("Reconnect folder");
    }
  } catch {
    setWorkingFolderStatus("No folder");
  }
}

function normalizeResult(result, groupId = state.groupId) {
  if (!result || typeof result !== "object" || !result.trial_id) return null;
  const { subject_id: subjectId, ...rest } = result;
  return {
    ...rest,
    expert_id: String(rest.expert_id || subjectId || state.expertId).trim() || state.expertId,
    group_id: groupId,
    pair_id: rest.pair_id || canonicalPairId(
      rest.dataset,
      rest.scene_id,
      rest.candidate_a_method,
      rest.candidate_b_method,
    ),
  };
}

function collectLocalResultsForGroup(groupId) {
  const collected = [];
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key?.startsWith("eciqa-pref:")) continue;
    try {
      const payload = JSON.parse(localStorage.getItem(key));
      if (!Array.isArray(payload?.results)) continue;
      for (const result of payload.results) {
        const resultGroup = result?.group_id || payload.group_id || "A";
        if (resultGroup !== groupId) continue;
        const normalized = normalizeResult(result, groupId);
        if (normalized) collected.push(normalized);
      }
    } catch {
      // Ignore unrelated or malformed localStorage entries.
    }
  }
  return collected;
}

async function comparedSceneIdsAcrossGroups(currentGroupResults) {
  const sceneIds = new Set();
  for (const groupId of GROUP_IDS) {
    const results = groupId === state.groupId
      ? currentGroupResults
      : await state.groupRepository.readGroup(groupId);
    for (const result of results.values()) {
      if (result.scene_id) sceneIds.add(result.scene_id);
    }
  }
  return sceneIds;
}

function readStoredResults(key, groupId) {
  const raw = localStorage.getItem(key);
  if (!raw) return { valid: false, results: [] };
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.results)) return { valid: false, results: [] };
    return {
      valid: true,
      results: parsed.results
        .map((result) => normalizeResult(result, groupId))
        .filter(Boolean),
    };
  } catch {
    return { valid: false, results: [] };
  }
}

function loadResults() {
  state.results = new Map();

  const legacy = state.groupId === "A"
    ? readStoredResults(state.legacyStorageKey, "A")
    : { valid: false, results: [] };
  const current = readStoredResults(state.storageKey, state.groupId);

  for (const result of legacy.results) {
    state.results.set(result.trial_id, result);
  }
  for (const result of current.results) {
    state.results.set(result.trial_id, result);
  }

  if (legacy.valid) {
    try {
      persistResults();
      localStorage.removeItem(state.legacyStorageKey);
    } catch {
      // Keep the legacy key intact if the browser cannot persist the migrated data.
    }
  }
}

function persistResults() {
  const payload = {
    expert_id: state.expertId,
    group_id: state.groupId,
    saved_at: new Date().toISOString(),
    results: Array.from(state.results.values()),
  };
  localStorage.setItem(state.storageKey, JSON.stringify(payload));
}

function hasValidDimensions(record) {
  return Number(record?.width) > 0 && Number(record?.height) > 0;
}

function applyLayout(record) {
  const orientation = imageOrientation(record);
  els.comparisonWorkspace.classList.toggle("layout-landscape", orientation === "landscape");
  els.comparisonWorkspace.classList.toggle("layout-portrait", orientation === "portrait");
}

function sizeImageFrame(slot, frame, image, record) {
  const slotRect = slot.getBoundingClientRect();
  const useNaturalSize = image.complete && image.naturalWidth > 0 && image.naturalHeight > 0;
  const sourceWidth = useNaturalSize ? image.naturalWidth : Number(record?.width);
  const sourceHeight = useNaturalSize ? image.naturalHeight : Number(record?.height);
  if (slotRect.width <= 0 || slotRect.height <= 0 || sourceWidth <= 0 || sourceHeight <= 0) return;

  const scale = Math.min(slotRect.width / sourceWidth, slotRect.height / sourceHeight);
  frame.style.width = `${Math.max(1, Math.floor(sourceWidth * scale))}px`;
  frame.style.height = `${Math.max(1, Math.floor(sourceHeight * scale))}px`;
}

function sizeImageFrames() {
  const trial = currentTrial();
  if (!trial) return;
  sizeImageFrame(els.inputSlot, els.inputFrame, els.inputImage, trial.input);
  sizeImageFrame(els.aSlot, els.aFrame, els.aImage, trial.a);
  sizeImageFrame(els.bSlot, els.bFrame, els.bImage, trial.b);
}

let pendingFrameSizing = 0;

function requestImageFrameSizing() {
  if (pendingFrameSizing) cancelAnimationFrame(pendingFrameSizing);
  pendingFrameSizing = requestAnimationFrame(() => {
    pendingFrameSizing = 0;
    sizeImageFrames();
  });
}

function clearChoiceButtons() {
  document.querySelectorAll(".choiceRow button").forEach((button) => {
    button.classList.remove("selected");
  });
}

function applyChoiceButtons() {
  clearChoiceButtons();
  for (const [key, value] of Object.entries(state.choices)) {
    const button = document.querySelector(`.choiceRow[data-key="${key}"] button[data-value="${value}"]`);
    if (button) button.classList.add("selected");
  }
}

function allChoicesReady() {
  return ["input_vs_a", "input_vs_b", "a_vs_b"].every((key) => state.choices[key]);
}

function updateRanking() {
  const box = document.querySelector(".rankingBox");
  box.classList.remove("conflict");
  if (!allChoicesReady()) {
    els.rankingText.textContent = "Choose all three comparisons";
    els.nextBtn.disabled = true;
    return;
  }
  const summary = summarizeRanking(state.choices);
  if (summary.hasConflict) {
    box.classList.add("conflict");
    els.rankingText.textContent = "Conflict: revise one comparison";
    els.nextBtn.disabled = true;
    return;
  }
  els.rankingText.textContent = summary.text;
  els.nextBtn.disabled = state.writePending;
}

function currentTrial() {
  return state.trials[state.index] || null;
}

function updateComparisonMarkButton() {
  const trial = currentTrial();
  if (!trial) {
    els.comparisonMarkBtn.disabled = true;
    els.comparisonMarkBtn.textContent = "Mark Comparison";
    els.comparisonMarkBtn.classList.remove("marked");
    els.comparisonMarkBtn.setAttribute("aria-pressed", "false");
    els.comparisonMarkBtn.removeAttribute("title");
    return;
  }

  const marked = isComparisonMarked(state.comparisonMarks, trial.pair_id);
  els.comparisonMarkBtn.disabled = state.markWritePending;
  els.comparisonMarkBtn.textContent = marked ? "Unmark Comparison" : "Mark Comparison";
  els.comparisonMarkBtn.classList.toggle("marked", marked);
  els.comparisonMarkBtn.setAttribute("aria-pressed", String(marked));
  els.comparisonMarkBtn.title = marked
    ? "Remove this comparison from the marked list"
    : "Add this comparison to the marked list";
}

function displayTrial() {
  const trial = currentTrial();
  if (!trial) {
    delete els.comparisonWorkspace.dataset.pairId;
    els.progressText.textContent = state.emptyMessage;
    els.sceneText.textContent = "";
    els.nextBtn.disabled = true;
    els.prevBtn.disabled = true;
    updateComparisonMarkButton();
    return;
  }

  const saved = state.results.get(trial.trial_id) || state.groupResults.get(trial.pair_id);
  state.choices = choicesForTrial(saved, trial);
  state.trialStartedAt = Date.now();

  els.comparisonWorkspace.dataset.pairId = trial.pair_id;
  applyLayout(trial.input);
  els.inputImage.src = trial.input.path;
  els.aImage.src = trial.a.path;
  els.bImage.src = trial.b.path;
  sizeImageFrames();

  const savedCount = state.results.size;
  els.progressText.textContent = `Trial ${state.index + 1}/${state.trials.length} | saved ${savedCount}`;
  els.sceneText.textContent = `${trial.dataset} | ${trial.scene_id}`;
  els.prevBtn.disabled = state.index === 0;
  els.exportCsvBtn.disabled = state.results.size === 0;
  els.exportJsonBtn.disabled = state.results.size === 0;
  els.clearBtn.disabled = state.results.size === 0;
  els.nextBtn.textContent = state.index === state.trials.length - 1 ? "Save" : "Save & Next";

  applyChoiceButtons();
  updateRanking();
  updateComparisonMarkButton();
}

function makeResult() {
  const trial = currentTrial();
  const ranking = summarizeRanking(state.choices);
  return {
    expert_id: state.expertId,
    group_id: state.groupId,
    pair_id: trial.pair_id,
    trial_id: trial.trial_id,
    dataset: trial.dataset,
    scene_id: trial.scene_id,
    input_method: trial.input.method,
    candidate_a_method: trial.a.method,
    candidate_b_method: trial.b.method,
    input_path: trial.input.path,
    candidate_a_path: trial.a.path,
    candidate_b_path: trial.b.path,
    input_vs_a: state.choices.input_vs_a,
    input_vs_b: state.choices.input_vs_b,
    a_vs_b: state.choices.a_vs_b,
    ranking: ranking.text,
    elapsed_ms: Date.now() - state.trialStartedAt,
    timestamp: new Date().toISOString(),
  };
}

async function saveCurrent() {
  if (!allChoicesReady()) return false;
  const ranking = summarizeRanking(state.choices);
  if (ranking.hasConflict) return false;
  const result = makeResult();
  state.writePending = true;
  updateRanking();
  try {
    await state.groupRepository.saveResult(state.groupId, result);
    state.groupResults.set(result.pair_id, result);
    state.results.set(result.trial_id, result);
    persistResults();
    setWorkingFolderStatus(`Folder: ${state.directoryHandle.name || "selected"}`, "connected");
    return true;
  } catch (error) {
    setWorkingFolderStatus(`Write failed: ${error.message}`, "error");
    return false;
  } finally {
    state.writePending = false;
    updateRanking();
  }
}

async function toggleCurrentComparisonMark() {
  const trial = currentTrial();
  if (!trial || !state.groupRepository || state.markWritePending) return;
  const currentlyMarked = isComparisonMarked(state.comparisonMarks, trial.pair_id);

  state.markWritePending = true;
  updateComparisonMarkButton();
  try {
    const update = await state.groupRepository.setComparisonMarked({
      expert_id: state.expertId,
      group_id: state.groupId,
      trial,
      marked: !currentlyMarked,
      marked_at: new Date().toISOString(),
    });
    state.comparisonMarks = update.marks;
    const action = update.marked ? "Marked" : "Unmarked";
    setWorkingFolderStatus(
      `${action}: ${trial.scene_id} comparison`,
      "connected",
    );
  } catch (error) {
    setWorkingFolderStatus(`Mark failed: ${error.message}`, "error");
  } finally {
    state.markWritePending = false;
    updateComparisonMarkButton();
  }
}

async function startSession() {
  if (!state.groupRepository) {
    setWorkingFolderStatus("Choose a working folder", "error");
    return;
  }
  els.startBtn.disabled = true;
  state.expertId = safeExpert();
  state.groupId = safeGroup();
  state.storageKey = storageKey(state.expertId, state.groupId);
  state.legacyStorageKey = legacyStorageKey(state.expertId);
  try {
    loadResults();
    state.comparisonMarks = await state.groupRepository.readComparisonMarks();
    const markedOnly = els.pairMode.value === "marked";
    const colorFixCoverage = els.pairMode.value === COLOR_FIX_COVERAGE_MODE;
    const trialOptions = {
      dataset: els.datasetFilter.value,
      pairMode: els.pairMode.value,
      maxTrials: 0,
      seed: safeSeed(),
    };
    let allTrials = markedOnly
      ? comparisonMarksToTrials(state.comparisonMarks, records)
          .filter((trial) =>
            els.datasetFilter.value === "all" || trial.dataset === els.datasetFilter.value)
      : buildTrials(scenes, trialOptions);
    const eligiblePairIds = new Set(allTrials.map((trial) => trial.pair_id));
    const localResults = collectLocalResultsForGroup(state.groupId)
      .filter((result) => eligiblePairIds.has(result.pair_id));
    const merged = await state.groupRepository.mergeResults(state.groupId, localResults);
    state.groupResults = merged.results;
    if (colorFixCoverage) {
      const seenSceneIds = await comparedSceneIdsAcrossGroups(state.groupResults);
      allTrials = buildTrials(scenes, { ...trialOptions, seenSceneIds });
    }

    state.trials = selectUnfinishedTrials(
      allTrials,
      state.groupResults,
      Number(els.maxTrials.value),
    );
    state.results = new Map();
    state.index = 0;
    state.emptyMessage = markedOnly
      ? `No unfinished marked comparisons for Group ${state.groupId}`
      : colorFixCoverage
        ? `No unfinished color-fix coverage trials for Group ${state.groupId}`
        : `No unfinished trials for Group ${state.groupId}`;
    els.exportCsvBtn.disabled = true;
    els.exportJsonBtn.disabled = true;
    els.clearBtn.disabled = true;
    setWorkingFolderStatus(`Folder: ${state.directoryHandle.name || "selected"}`, "connected");
    displayTrial();
  } catch (error) {
    state.trials = [];
    state.results = new Map();
    state.emptyMessage = "Could not start session";
    setWorkingFolderStatus(`Read failed: ${error.message}`, "error");
    displayTrial();
  } finally {
    els.startBtn.disabled = false;
  }
}

function download(filename, mimeType, text) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportCsv() {
  const rows = state.trials
    .map((trial) => state.results.get(trial.trial_id))
    .filter(Boolean)
    .map((result) => resultToCsvRow(result));
  const csv = [CSV_HEADER.join(","), ...rows].join("\r\n");
  download(`${state.expertId}_group_${state.groupId}_mos_results.csv`, "text/csv;charset=utf-8", csv);
}

function exportJson() {
  const payload = {
    expert_id: state.expertId,
    group_id: state.groupId,
    exported_at: new Date().toISOString(),
    storage_key: state.storageKey,
    result_count: state.results.size,
    results: state.trials.map((trial) => state.results.get(trial.trial_id)).filter(Boolean),
  };
  download(
    `${state.expertId}_group_${state.groupId}_mos_results.json`,
    "application/json;charset=utf-8",
    JSON.stringify(payload, null, 2),
  );
}

function resetSaved() {
  if (!state.storageKey) return;
  const ok = window.confirm("Clear this browser session? Group CSV records will remain.");
  if (!ok) return;
  localStorage.removeItem(state.storageKey);
  state.results = new Map();
  displayTrial();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function hideLenses() {
  document.querySelectorAll(".lens").forEach((lens) => lens.classList.remove("active"));
}

function updateLens(img, lens, relX, relY) {
  if (!img.complete || !img.currentSrc) return;
  const imgRect = img.getBoundingClientRect();
  const boxRect = img.parentElement.getBoundingClientRect();
  const size = 190;
  const zoom = 3;
  const x = imgRect.left - boxRect.left + relX * imgRect.width;
  const y = imgRect.top - boxRect.top + relY * imgRect.height;
  const left = clamp(x - size / 2, 0, Math.max(0, boxRect.width - size));
  const top = clamp(y - size / 2, 0, Math.max(0, boxRect.height - size));

  lens.style.left = `${left}px`;
  lens.style.top = `${top}px`;
  lens.style.backgroundImage = `url("${img.currentSrc}")`;
  lens.style.backgroundSize = `${imgRect.width * zoom}px ${imgRect.height * zoom}px`;
  lens.style.backgroundPosition = `${size / 2 - relX * imgRect.width * zoom}px ${size / 2 - relY * imgRect.height * zoom}px`;
  lens.classList.add("active");
}

function setupZoom() {
  const imagePairs = [
    [els.inputImage, document.getElementById("lensInput")],
    [els.aImage, document.getElementById("lensA")],
    [els.bImage, document.getElementById("lensB")],
  ];

  els.imageGrid.addEventListener("mousemove", (event) => {
    const hovered = imagePairs.map(([img]) => img).find((img) => {
      const rect = img.getBoundingClientRect();
      return event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;
    });
    if (!hovered) {
      hideLenses();
      return;
    }
    const rect = hovered.getBoundingClientRect();
    const relX = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const relY = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    for (const [img, lens] of imagePairs) {
      updateLens(img, lens, relX, relY);
    }
  });

  els.imageGrid.addEventListener("mouseleave", hideLenses);
}

document.querySelectorAll(".choiceRow button").forEach((button) => {
  button.addEventListener("click", () => {
    const row = button.closest(".choiceRow");
    state.choices[row.dataset.key] = button.dataset.value;
    applyChoiceButtons();
    updateRanking();
  });
});

els.startBtn.addEventListener("click", startSession);
els.workingFolderBtn.addEventListener("click", chooseWorkingFolder);
els.comparisonMarkBtn.addEventListener("click", toggleCurrentComparisonMark);
els.prevBtn.addEventListener("click", () => {
  if (state.index > 0) {
    state.index -= 1;
    displayTrial();
  }
});
els.nextBtn.addEventListener("click", async () => {
  if (!await saveCurrent()) return;
  if (state.index < state.trials.length - 1) {
    state.index += 1;
    displayTrial();
  } else {
    displayTrial();
  }
});
els.exportCsvBtn.addEventListener("click", exportCsv);
els.exportJsonBtn.addEventListener("click", exportJson);
els.clearBtn.addEventListener("click", resetSaved);
els.inputImage.addEventListener("load", () => {
  const trial = currentTrial();
  if (!trial) return;
  if (!hasValidDimensions(trial.input)) {
    applyLayout({
      width: els.inputImage.naturalWidth,
      height: els.inputImage.naturalHeight,
    });
  }
  requestImageFrameSizing();
});
els.aImage.addEventListener("load", requestImageFrameSizing);
els.bImage.addEventListener("load", requestImageFrameSizing);
window.addEventListener("resize", requestImageFrameSizing);

setMetaText();
setupZoom();
restoreWorkingFolder();
