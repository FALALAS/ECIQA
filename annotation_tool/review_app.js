const {
  imageOrientation,
  summarizeRanking,
} = window.MosCore;
const {
  GROUP_IDS,
  annotationReviewId,
  createGroupCsvRepository,
  isAnnotationReviewed,
  isComparisonMarked,
} = window.MosGroupCsv;

const records = window.MOS_RECORDS || [];
const recordsByPath = new Map(
  records.filter((record) => record?.path).map((record) => [record.path, record]),
);
const CHOICE_KEYS = ["input_vs_a", "input_vs_b", "a_vs_b"];

const els = {
  metaText: document.getElementById("metaText"),
  reviewerId: document.getElementById("reviewerId"),
  reviewGroupId: document.getElementById("reviewGroupId"),
  workingFolderBtn: document.getElementById("workingFolderBtn"),
  workingFolderStatus: document.getElementById("workingFolderStatus"),
  sourceGroupFilter: document.getElementById("sourceGroupFilter"),
  datasetFilter: document.getElementById("datasetFilter"),
  reviewStatusFilter: document.getElementById("reviewStatusFilter"),
  maxReviews: document.getElementById("maxReviews"),
  startBtn: document.getElementById("startBtn"),
  progressText: document.getElementById("progressText"),
  sceneText: document.getElementById("sceneText"),
  comparisonMarkBtn: document.getElementById("comparisonMarkBtn"),
  prevBtn: document.getElementById("prevBtn"),
  reviewedBtn: document.getElementById("reviewedBtn"),
  skipBtn: document.getElementById("skipBtn"),
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
  items: [],
  index: 0,
  choices: {},
  reviews: new Map(),
  comparisonMarks: new Map(),
  groupRepository: null,
  directoryHandle: null,
  rememberedDirectoryHandle: null,
  reviewerId: "anonymous",
  reviewGroupId: "A",
  writePending: false,
  markWritePending: false,
  emptyMessage: "No annotations loaded",
};

const HANDLE_DB_NAME = "eciqa-pref-working-folder";
const HANDLE_STORE_NAME = "handles";
const HANDLE_KEY = "working-folder";

function safeReviewer() {
  return (els.reviewerId.value || "anonymous").trim() || "anonymous";
}

function safeReviewGroup() {
  const groupId = String(els.reviewGroupId.value || "A").trim().toUpperCase();
  return GROUP_IDS.includes(groupId) ? groupId : "A";
}

function setMetaText() {
  els.metaText.textContent = records.length
    ? `${records.length} images | saved annotations from Group A-E`
    : "manifest.js is missing or empty";
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
      // The selected folder remains usable even if the browser cannot remember it.
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

function recordFor(path, method, itemType) {
  return recordsByPath.get(path) || {
    path,
    method,
    item_type: itemType,
  };
}

function sourceResultToItem(result) {
  const annotationId = annotationReviewId(result);
  if (!annotationId || !result?.input_path || !result?.candidate_a_path ||
      !result?.candidate_b_path) return null;
  return {
    annotation_id: annotationId,
    source_result: result,
    pair_id: result.pair_id,
    trial_id: result.trial_id,
    dataset: result.dataset,
    scene_id: result.scene_id,
    input: recordFor(result.input_path, result.input_method || "input", "input"),
    a: recordFor(result.candidate_a_path, result.candidate_a_method, "candidate"),
    b: recordFor(result.candidate_b_path, result.candidate_b_method, "candidate"),
  };
}

async function readSourceItems() {
  const sourceFilter = String(els.sourceGroupFilter.value || "all");
  const sourceGroups = sourceFilter === "all" ? GROUP_IDS : [sourceFilter];
  const datasetFilter = String(els.datasetFilter.value || "all");
  const items = [];
  for (const sourceGroupId of sourceGroups) {
    const results = await state.groupRepository.readGroup(sourceGroupId);
    for (const result of results.values()) {
      if (datasetFilter !== "all" && result.dataset !== datasetFilter) continue;
      const item = sourceResultToItem(result);
      if (item) items.push(item);
    }
  }
  return items;
}

function selectReviewItems(items) {
  const status = String(els.reviewStatusFilter.value || "pending");
  const filtered = items.filter((item) => {
    const reviewed = isAnnotationReviewed(state.reviews, item.annotation_id);
    if (status === "reviewed") return reviewed;
    if (status === "pending") return !reviewed;
    return true;
  });
  const maximum = Number(els.maxReviews.value);
  return Number.isFinite(maximum) && maximum > 0 ? filtered.slice(0, maximum) : filtered;
}

function currentItem() {
  return state.items[state.index] || null;
}

function currentTrial() {
  const item = currentItem();
  if (!item) return null;
  return {
    pair_id: item.pair_id,
    trial_id: item.trial_id,
    dataset: item.dataset,
    scene_id: item.scene_id,
    input: item.input,
    a: item.a,
    b: item.b,
  };
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
  const item = currentItem();
  if (!item) return;
  sizeImageFrame(els.inputSlot, els.inputFrame, els.inputImage, item.input);
  sizeImageFrame(els.aSlot, els.aFrame, els.aImage, item.a);
  sizeImageFrame(els.bSlot, els.bFrame, els.bImage, item.b);
}

let pendingFrameSizing = 0;

function requestImageFrameSizing() {
  if (pendingFrameSizing) cancelAnimationFrame(pendingFrameSizing);
  pendingFrameSizing = requestAnimationFrame(() => {
    pendingFrameSizing = 0;
    sizeImageFrames();
  });
}

function choicesFrom(result) {
  return Object.fromEntries(CHOICE_KEYS.map((key) => [key, result?.[key] || ""]));
}

function applyChoiceButtons() {
  document.querySelectorAll(".choiceRow button").forEach((button) => {
    button.classList.remove("selected");
  });
  for (const key of CHOICE_KEYS) {
    const value = state.choices[key];
    const button = document.querySelector(
      `.choiceRow[data-key="${key}"] button[data-value="${value}"]`,
    );
    if (button) button.classList.add("selected");
  }
}

function currentRanking() {
  return summarizeRanking(state.choices);
}

function allChoicesReady() {
  return CHOICE_KEYS.every((key) => state.choices[key]);
}

function choicesMatchReview(review) {
  return Boolean(review) && CHOICE_KEYS.every((key) => review[key] === state.choices[key]);
}

function updateRanking() {
  const rankingBox = document.querySelector(".rankingBox");
  rankingBox.classList.remove("conflict");
  if (!allChoicesReady()) {
    els.rankingText.textContent = "Choose all three comparisons";
    updateReviewedButton();
    return;
  }
  const summary = currentRanking();
  if (summary.hasConflict) {
    rankingBox.classList.add("conflict");
    els.rankingText.textContent = "Conflict: revise one comparison";
  } else {
    els.rankingText.textContent = summary.text;
  }
  updateReviewedButton();
}

function updateProgress() {
  const item = currentItem();
  if (!item) return;
  els.progressText.textContent = [
    `Review ${state.index + 1}/${state.items.length}`,
    `reviewed ${state.reviews.size}`,
    `source ${item.source_result.group_id}`,
  ].join(" | ");
}

function updateReviewedButton() {
  const item = currentItem();
  if (!item) {
    els.reviewedBtn.disabled = true;
    els.reviewedBtn.classList.remove("reviewed");
    els.reviewedBtn.setAttribute("aria-pressed", "false");
    els.reviewedBtn.removeAttribute("title");
    return;
  }
  const storedReview = state.reviews.get(item.annotation_id);
  const matchesStoredReview = choicesMatchReview(storedReview);
  const summary = currentRanking();
  els.reviewedBtn.disabled = state.writePending || !allChoicesReady() || summary.hasConflict;
  els.reviewedBtn.classList.toggle("reviewed", matchesStoredReview);
  els.reviewedBtn.setAttribute("aria-pressed", String(matchesStoredReview));
  els.reviewedBtn.title = matchesStoredReview
    ? "Remove the reviewed state for this source annotation"
    : storedReview
      ? "Save the updated reviewed choices"
      : "Save these choices as the reviewed annotation";
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
    ? "Remove this comparison from the shared marked list"
    : "Add this comparison to the shared marked list";
}

function displayItem() {
  const item = currentItem();
  if (!item) {
    delete els.comparisonWorkspace.dataset.pairId;
    delete els.comparisonWorkspace.dataset.annotationId;
    els.progressText.textContent = state.emptyMessage;
    els.sceneText.textContent = "";
    els.rankingText.textContent = "No annotation loaded";
    state.choices = {};
    applyChoiceButtons();
    els.reviewedBtn.disabled = true;
    els.skipBtn.disabled = true;
    els.prevBtn.disabled = state.items.length === 0 || state.index === 0;
    updateReviewedButton();
    updateComparisonMarkButton();
    return;
  }

  els.comparisonWorkspace.dataset.pairId = item.pair_id;
  els.comparisonWorkspace.dataset.annotationId = item.annotation_id;
  applyLayout(item.input);
  els.inputImage.src = item.input.path;
  els.aImage.src = item.a.path;
  els.bImage.src = item.b.path;
  sizeImageFrames();

  const source = item.source_result;
  updateProgress();
  els.sceneText.textContent = [
    item.dataset,
    item.scene_id,
    `Source Group ${source.group_id}`,
    `Expert ${source.expert_id}`,
  ].join(" | ");
  els.prevBtn.disabled = state.index === 0;
  els.skipBtn.disabled = false;
  els.skipBtn.textContent = state.index === state.items.length - 1 ? "Finish" : "Skip";
  const storedReview = state.reviews.get(item.annotation_id);
  state.choices = choicesFrom(storedReview || source);
  applyChoiceButtons();
  updateRanking();
  updateComparisonMarkButton();
}

function makeReviewedResult(item) {
  const ranking = currentRanking();
  return {
    ...item.source_result,
    input_vs_a: state.choices.input_vs_a,
    input_vs_b: state.choices.input_vs_b,
    a_vs_b: state.choices.a_vs_b,
    ranking: ranking.text,
  };
}

async function toggleCurrentReviewed() {
  const item = currentItem();
  if (!item || !state.groupRepository || state.writePending) return;
  const storedReview = state.reviews.get(item.annotation_id);
  const matchesStoredReview = choicesMatchReview(storedReview);
  const shouldReview = !matchesStoredReview;
  const ranking = currentRanking();
  if (shouldReview && (!allChoicesReady() || ranking.hasConflict)) return;
  state.writePending = true;
  updateReviewedButton();
  try {
    const update = await state.groupRepository.setReviewed({
      reviewer_id: state.reviewerId,
      review_group_id: state.reviewGroupId,
      source_result: item.source_result,
      review_result: shouldReview ? makeReviewedResult(item) : undefined,
      reviewed: shouldReview,
      reviewed_at: new Date().toISOString(),
    });
    state.reviews = update.reviews;
    const action = update.reviewed
      ? storedReview ? "Review updated" : "Reviewed"
      : "Review removed";
    setWorkingFolderStatus(
      `${action}: Source Group ${item.source_result.group_id} | ${item.scene_id}`,
      "connected",
    );
  } catch (error) {
    setWorkingFolderStatus(`Review write failed: ${error.message}`, "error");
  } finally {
    state.writePending = false;
    updateProgress();
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
      expert_id: state.reviewerId,
      group_id: state.reviewGroupId,
      trial,
      marked: !currentlyMarked,
      marked_at: new Date().toISOString(),
    });
    state.comparisonMarks = update.marks;
    const action = update.marked ? "Marked" : "Unmarked";
    setWorkingFolderStatus(`${action}: ${trial.scene_id} comparison`, "connected");
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
  state.reviewerId = safeReviewer();
  state.reviewGroupId = safeReviewGroup();
  try {
    state.reviews = await state.groupRepository.readReviewGroup(state.reviewGroupId);
    state.comparisonMarks = await state.groupRepository.readComparisonMarks();
    const allItems = await readSourceItems();
    state.items = selectReviewItems(allItems);
    state.index = 0;
    const status = String(els.reviewStatusFilter.value || "pending");
    state.emptyMessage = status === "pending"
      ? `No pending annotations for Review Group ${state.reviewGroupId}`
      : status === "reviewed"
        ? `No reviewed annotations for Review Group ${state.reviewGroupId}`
        : "No source annotations match the filters";
    setWorkingFolderStatus(`Folder: ${state.directoryHandle.name || "selected"}`, "connected");
    displayItem();
  } catch (error) {
    state.items = [];
    state.index = 0;
    state.emptyMessage = "Could not start review session";
    setWorkingFolderStatus(`Read failed: ${error.message}`, "error");
    displayItem();
  } finally {
    els.startBtn.disabled = false;
  }
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
      return event.clientX >= rect.left && event.clientX <= rect.right &&
        event.clientY >= rect.top && event.clientY <= rect.bottom;
    });
    if (!hovered) {
      hideLenses();
      return;
    }
    const rect = hovered.getBoundingClientRect();
    const relX = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const relY = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    for (const [img, lens] of imagePairs) updateLens(img, lens, relX, relY);
  });
  els.imageGrid.addEventListener("mouseleave", hideLenses);
}

els.startBtn.addEventListener("click", startSession);
els.workingFolderBtn.addEventListener("click", chooseWorkingFolder);
els.comparisonMarkBtn.addEventListener("click", toggleCurrentComparisonMark);
els.reviewedBtn.addEventListener("click", toggleCurrentReviewed);
document.querySelectorAll(".choiceRow button").forEach((button) => {
  button.addEventListener("click", () => {
    if (state.writePending || !currentItem()) return;
    const row = button.closest(".choiceRow");
    state.choices[row.dataset.key] = button.dataset.value;
    applyChoiceButtons();
    updateRanking();
  });
});
els.prevBtn.addEventListener("click", () => {
  if (state.index > 0) {
    state.index = Math.min(state.index - 1, state.items.length - 1);
    displayItem();
  }
});
els.skipBtn.addEventListener("click", () => {
  if (!currentItem()) return;
  if (state.index < state.items.length - 1) {
    state.index += 1;
  } else {
    state.index = state.items.length;
    state.emptyMessage = "Review session complete";
  }
  displayItem();
});

els.inputImage.addEventListener("load", () => {
  const item = currentItem();
  if (!item) return;
  if (!hasValidDimensions(item.input)) {
    applyLayout({ width: els.inputImage.naturalWidth, height: els.inputImage.naturalHeight });
  }
  requestImageFrameSizing();
});
els.aImage.addEventListener("load", requestImageFrameSizing);
els.bImage.addEventListener("load", requestImageFrameSizing);
window.addEventListener("resize", requestImageFrameSizing);

setMetaText();
setupZoom();
restoreWorkingFolder();
