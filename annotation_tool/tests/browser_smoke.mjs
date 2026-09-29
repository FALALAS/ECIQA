import assert from "node:assert/strict";
import { promisify } from "node:util";
import { execFile } from "node:child_process";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
// Start serve.py first. The browser loads real dataset images, while the
// File System Access folder below is an in-memory test double: this smoke
// does not validate native directory permissions or real folder writes.
const appUrl = new URL(process.env.ECIQA_ANNOTATION_URL || "http://127.0.0.1:8010/").href;
const reviewUrl = new URL("review.html", appUrl).href;
const execFileAsync = promisify(execFile);

const browserCandidates = [
  process.env.ECIQA_BROWSER_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

const browserPath = browserCandidates.find((candidate) => candidate && fs.existsSync(candidate));
assert.ok(browserPath, "Set ECIQA_BROWSER_PATH to an installed Chrome, Chromium, or Edge executable");
assert.ok(typeof WebSocket === "function", "This Node.js runtime does not provide WebSocket");

const port = 9300 + Math.floor(Math.random() * 500);
const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "eciqa-label-smoke-"));
const screenshotDir = process.env.ECIQA_SCREENSHOT_DIR || userDataDir;
fs.mkdirSync(screenshotDir, { recursive: true });
let browserStderr = "";
const browser = spawn(browserPath, [
  "--headless=new",
  "--disable-gpu",
  "--force-device-scale-factor=1",
  "--window-size=1600,1000",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${userDataDir}`,
  appUrl,
], { stdio: ["ignore", "ignore", "pipe"], windowsHide: true });
browser.stderr.on("data", (chunk) => {
  browserStderr += chunk.toString();
});

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(fn, timeoutMs = 15000) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await sleep(200);
  }
  if (lastError) throw lastError;
  throw new Error("Timed out waiting for browser state");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

async function connectCdp(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  let socketError = "";
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let nextId = 1;
  const pending = new Map();
  const events = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.method === "Runtime.exceptionThrown") {
      events.push({
        method: message.method,
        text: message.params?.exceptionDetails?.text,
        description: message.params?.exceptionDetails?.exception?.description,
      });
    }
    if (message.method === "Log.entryAdded") {
      events.push({
        method: message.method,
        level: message.params?.entry?.level,
        text: message.params?.entry?.text,
      });
    }
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) {
      reject(new Error(`${message.error.message}: ${message.error.data || ""}`));
    } else {
      resolve(message.result);
    }
  });
  socket.addEventListener("close", (event) => {
    const error = new Error([
      `CDP socket closed (${event.code}): ${event.reason || "no reason"}`,
      socketError ? `Socket error: ${socketError}` : "",
      browserStderr ? `Browser stderr:\n${browserStderr}` : "",
    ].filter(Boolean).join("\n"));
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
  });
  socket.addEventListener("error", (event) => {
    socketError = event.message || event.error?.message || String(event.error || "unknown WebSocket error");
  });

  return {
    socket,
    events,
    send(method, params = {}) {
      const id = nextId;
      nextId += 1;
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`Timed out waiting for CDP response: ${method}`));
        }, 10000);
        pending.set(id, {
          resolve(value) {
            clearTimeout(timeout);
            resolve(value);
          },
          reject(error) {
            clearTimeout(timeout);
            reject(error);
          },
        });
        try {
          socket.send(JSON.stringify({ id, method, params }));
        } catch (error) {
          pending.delete(id);
          clearTimeout(timeout);
          reject(error);
        }
      });
    },
  };
}

let cdp = null;
try {
  const targets = await waitFor(async () => {
    const pages = await fetchJson(`http://127.0.0.1:${port}/json`);
    return pages.find((page) => page.type === "page" && page.url === appUrl);
  });
  cdp = await connectCdp(targets.webSocketDebuggerUrl);
  await cdp.send("Runtime.enable");
  await cdp.send("Page.enable");
  await cdp.send("Log.enable");

  async function evaluate(expression) {
    const response = await cdp.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (response.exceptionDetails) {
      throw new Error(response.exceptionDetails.text || "Runtime evaluation failed");
    }
    return response.result?.value;
  }

  async function layoutSnapshot() {
    return evaluate(`(() => {
      const workspace = document.getElementById("comparisonWorkspace");
      if (!workspace) return null;
      const elementRect = (element) => {
        if (!element) return null;
        const bounds = element.getBoundingClientRect();
        return {
          left: bounds.left,
          right: bounds.right,
          top: bounds.top,
          bottom: bounds.bottom,
          width: bounds.width,
          height: bounds.height,
        };
      };
      const rect = (selector) => elementRect(document.querySelector(selector));
      const frame = (slotSelector, panelSelector, imageSelector) => {
        const slot = document.querySelector(slotSelector);
        const panel = document.querySelector(panelSelector);
        const image = document.querySelector(imageSelector);
        const header = panel?.querySelector(".panelHead");
        return {
          slot: elementRect(slot),
          panel: elementRect(panel),
          image: elementRect(image),
          header: elementRect(header),
          headerPosition: header ? getComputedStyle(header).position : "",
          naturalWidth: image?.naturalWidth || 0,
          naturalHeight: image?.naturalHeight || 0,
        };
      };
      return {
        className: workspace.className,
        workspace: rect("#comparisonWorkspace"),
        judgement: rect(".judgement"),
        judgementOverflow: (() => {
          const panel = document.querySelector(".judgement");
          return {
            clientHeight: panel.clientHeight,
            scrollHeight: panel.scrollHeight,
            overflowY: getComputedStyle(panel).overflowY,
          };
        })(),
        input: rect('[data-label="Input"]'),
        a: rect('[data-label="A"]'),
        b: rect('[data-label="B"]'),
        inputBox: rect(".inputPanel .imageBox"),
        inputImage: rect("#inputImage"),
        frames: {
          input: frame(".inputPanel", '[data-label="Input"]', "#inputImage"),
          a: frame(".candidateAPanel", '[data-label="A"]', "#aImage"),
          b: frame(".candidateBPanel", '[data-label="B"]', "#bImage"),
        },
        chromeInsideJudgement: [".topbar", ".statusbar"].every((selector) =>
          document.querySelector(".judgement")?.contains(document.querySelector(selector))),
        resolutionLabelCount: ["inputSize", "aSize", "bSize"].filter((id) =>
          document.getElementById(id)).length,
        labels: Array.from(document.querySelectorAll(".panelHead strong"), (node) => node.textContent.trim()),
      };
    })()`);
  }

  function overlaps(first, second) {
    return first.left < second.right - 1 &&
      first.right > second.left + 1 &&
      first.top < second.bottom - 1 &&
      first.bottom > second.top + 1;
  }

  function assertJudgementOverflow(layout, label) {
    const panel = layout.judgementOverflow;
    if (panel.scrollHeight > panel.clientHeight + 1) {
      assert.equal(panel.overflowY, "auto",
        `${label}: overflowing controls must scroll inside their panel, not cover an image`);
    }
  }

  function assertAdaptiveFrame(frame, label) {
    assert.equal(frame.headerPosition, "absolute", `${label} label must overlay the image`);
    assert.ok(frame.header.width < frame.panel.width * 0.5, `${label} label must not occupy a full row`);
    assert.ok(frame.header.left <= frame.panel.left + 16, `${label} label must stay at the top-left`);
    assert.ok(frame.header.top <= frame.panel.top + 16, `${label} label must stay at the top-left`);

    for (const edge of ["left", "right", "top", "bottom"]) {
      assert.ok(
        Math.abs(frame.image[edge] - frame.panel[edge]) <= 4,
        `${label} image must fill its aspect-ratio frame on ${edge}`,
      );
    }

    const panelAspect = frame.panel.width / frame.panel.height;
    const imageAspect = frame.naturalWidth / frame.naturalHeight;
    assert.ok(Math.abs(panelAspect - imageAspect) < 0.02, `${label} frame must follow the image aspect ratio`);
    assert.ok(
      Math.abs(frame.panel.width - frame.slot.width) <= 4 ||
        Math.abs(frame.panel.height - frame.slot.height) <= 4,
      `${label} frame must maximize one available dimension`,
    );
  }

  async function captureScreenshot(name) {
    const { data } = await cdp.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
    });
    const destination = path.join(screenshotDir, name);
    fs.writeFileSync(destination, data, "base64");
    assert.ok(fs.statSync(destination).size > 10000, `${name} should contain a rendered page`);
    return destination;
  }

  let metaText = "";
  try {
    metaText = await waitFor(async () => {
      const text = await evaluate("document.getElementById('metaText')?.textContent || ''");
      return text.includes("images") ? text : "";
    }, 30000);
  } catch (error) {
    const diagnostics = await evaluate(`JSON.stringify({
      readyState: document.readyState,
      metaText: document.getElementById("metaText")?.textContent || "",
      recordCount: Array.isArray(window.MOS_RECORDS) ? window.MOS_RECORDS.length : null,
      hasCore: Boolean(window.MosCore),
      scripts: Array.from(document.scripts).map((script) => ({ src: script.src, type: script.type }))
    })`);
    console.error(`Diagnostics: ${diagnostics}`);
    console.error(`Browser events: ${JSON.stringify(cdp.events, null, 2)}`);
    throw error;
  }
  assert.match(metaText, /5200 images/);

  const identityFields = await evaluate(`(() => {
    const expert = document.getElementById("expertId");
    const expertLabel = document.querySelector('label[for="expertId"]');
    const group = document.getElementById("groupId");
    const groupLabel = document.querySelector('label[for="groupId"]');
    const pairMode = document.getElementById("pairMode");
    return {
      expertExists: Boolean(expert),
      expertLabel: expertLabel?.childNodes[0]?.textContent.trim() || "",
      groupExists: Boolean(group),
      groupLabel: groupLabel?.childNodes[0]?.textContent.trim() || "",
      groupValue: group?.value || "",
      groupOptions: Array.from(group?.options || [], (option) => option.value),
      pairModeOptions: Array.from(pairMode?.options || [], (option) => option.value),
    };
  })()`);
  assert.deepEqual(identityFields, {
    expertExists: true,
    expertLabel: "Expert ID",
    groupExists: true,
    groupLabel: "Group",
    groupValue: "A",
    groupOptions: ["A", "B", "C", "D", "E"],
    pairModeOptions: ["compact", "unseen", "all", "marked"],
  });
  assert.equal(
    await evaluate(`document.getElementById("clearBtn")?.textContent.trim() || ""`),
    "Clear Session",
  );

  const disconnectedFolder = await evaluate(`(() => ({
    buttonExists: Boolean(document.getElementById("workingFolderBtn")),
    statusText: document.getElementById("workingFolderStatus")?.textContent.trim() || "",
    startDisabled: Boolean(document.getElementById("startBtn")?.disabled),
    comparisonMarkExists: Boolean(document.getElementById("comparisonMarkBtn")),
    comparisonMarkDisabled: Boolean(document.getElementById("comparisonMarkBtn")?.disabled),
  }))()`);
  assert.deepEqual(disconnectedFolder, {
    buttonExists: true,
    statusText: "No folder",
    startDisabled: true,
    comparisonMarkExists: true,
    comparisonMarkDisabled: true,
  });

  const folderFixture = `(() => {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const files = new Map((window.__smokeFolderFiles || [])
      .map(([name, bytes]) => [name, new Uint8Array(bytes)]));
    const directory = {
      kind: "directory",
      name: "handoff-folder",
      files,
      async queryPermission() { return "granted"; },
      async requestPermission() { return "granted"; },
      async getFileHandle(name, options = {}) {
        if (!files.has(name)) {
          if (!options.create) throw new DOMException("Missing file", "NotFoundError");
          files.set(name, new Uint8Array());
        }
        return {
          kind: "file",
          name,
          async getFile() {
            const snapshot = files.get(name).slice();
            return {
              size: snapshot.byteLength,
              async text() { return decoder.decode(snapshot); },
            };
          },
          async createWritable(options = {}) {
            let content = options.keepExistingData
              ? files.get(name).slice()
              : new Uint8Array();
            let position = 0;
            const ensureSize = (size) => {
              if (size <= content.byteLength) return;
              const expanded = new Uint8Array(size);
              expanded.set(content);
              content = expanded;
            };
            return {
              async seek(nextPosition) { position = Number(nextPosition); },
              async write(value) {
                const incoming = value instanceof Uint8Array ? value : encoder.encode(String(value));
                ensureSize(position + incoming.byteLength);
                content.set(incoming, position);
                position += incoming.byteLength;
              },
              async close() { files.set(name, content); },
              async abort() {},
            };
          },
        };
      },
    };
    window.__mosFakeDirectory = directory;
    window.__mosPickerCalls = 0;
    window.showDirectoryPicker = async () => {
      window.__mosPickerCalls += 1;
      return directory;
    };
    document.getElementById("workingFolderBtn").click();
  })()`;
  await evaluate(folderFixture);

  const connectedFolder = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      statusText: document.getElementById("workingFolderStatus")?.textContent.trim() || "",
      startDisabled: Boolean(document.getElementById("startBtn")?.disabled),
      filenames: Array.from(window.__mosFakeDirectory?.files?.keys() || []).sort(),
    }))()`);
    return value.statusText.includes("handoff-folder") && value.filenames.length === 11 ? value : null;
  });
  assert.deepEqual(connectedFolder, {
    statusText: "Folder: handoff-folder",
    startDisabled: false,
    filenames: [
      "group_A.csv",
      "group_A_reviewed.csv",
      "group_B.csv",
      "group_B_reviewed.csv",
      "group_C.csv",
      "group_C_reviewed.csv",
      "group_D.csv",
      "group_D_reviewed.csv",
      "group_E.csv",
      "group_E_reviewed.csv",
      "marked_comparisons.csv",
    ],
  });
  assert.equal(
    await evaluate(`Array.from(window.__mosFakeDirectory.files.entries())
      .filter(([name]) => /^group_[A-E]\.csv$/.test(name))
      .map(([, bytes]) => new TextDecoder().decode(bytes).startsWith("expert_id,group_id,pair_id,trial_id"))
      .every(Boolean)`),
    true,
  );
  assert.equal(
    await evaluate(`new TextDecoder().decode(window.__mosFakeDirectory.files.get("marked_comparisons.csv"))
      .startsWith("mark_order,expert_id,group_id,pair_id,trial_id,dataset,scene_id,input_method,candidate_a_method,candidate_b_method,input_path,candidate_a_path,candidate_b_path,marked_at")`),
    true,
  );
  await evaluate(`document.getElementById("workingFolderBtn").click()`);
  await waitFor(async () => await evaluate(`!document.getElementById("workingFolderBtn").disabled`));
  assert.equal(
    await evaluate(`window.__mosPickerCalls`),
    2,
    "Change Folder must open the picker instead of silently reconnecting the current folder",
  );

  const legacySetup = await evaluate(`(() => {
    const trial = window.MosCore.buildTrials(window.MosCore.buildScenes(window.MOS_RECORDS), {
      dataset: "all",
      pairMode: "compact",
      maxTrials: 3,
      seed: "scene-order",
    })[0];
    const legacyKey = "eciqa-pref:smoke:all:compact:3:scene-order";
    localStorage.setItem(legacyKey, JSON.stringify({
      expert_id: "smoke",
      saved_at: "2026-07-29T00:00:00.000Z",
      results: [{
        expert_id: "smoke",
        trial_id: trial.trial_id,
        dataset: trial.dataset,
        scene_id: trial.scene_id,
        input_method: trial.input.method,
        candidate_a_method: trial.a.method,
        candidate_b_method: trial.b.method,
        input_path: trial.input.path,
        candidate_a_path: trial.a.path,
        candidate_b_path: trial.b.path,
        input_vs_a: "left",
        input_vs_b: "left",
        a_vs_b: "similar",
        ranking: "Input>A=B",
        elapsed_ms: 1000,
        timestamp: "2026-07-29T00:00:00.000Z",
      }],
    }));
    localStorage.setItem("eciqa-pref:stale:A:all:compact:1:old-seed", JSON.stringify({
      expert_id: "stale",
      group_id: "A",
      results: [{
        expert_id: "stale",
        group_id: "A",
        pair_id: "MSEC::removed_smoke_scene::cotf_mefssim_0.5x::cotf_original",
        trial_id: "removed_smoke_scene__cotf_mefssim_0.5x__vs__cotf_original",
        dataset: "MSEC",
        scene_id: "removed_smoke_scene",
        input_method: "input",
        candidate_a_method: "cotf_mefssim_0.5x",
        candidate_b_method: "cotf_original",
        input_path: "/images/scenes/removed_smoke_scene/00_input.jpg",
        candidate_a_path: "/images/scenes/removed_smoke_scene/11_cotf_mefssim_0.5x.png",
        candidate_b_path: "/images/scenes/removed_smoke_scene/10_cotf_original.png",
        input_vs_a: "left",
        input_vs_b: "left",
        a_vs_b: "similar",
        ranking: "Input>A=B",
        elapsed_ms: 900,
        timestamp: "2026-07-28T00:00:00.000Z",
      }],
    }));
    return { legacyKey, trialId: trial.trial_id };
  })()`);
  assert.ok(legacySetup.trialId);

  await evaluate(`
    document.getElementById("expertId").value = "smoke";
    document.getElementById("groupId").value = "A";
    document.getElementById("maxTrials").value = "3";
    document.getElementById("seedText").value = "scene-order";
    document.getElementById("startBtn").click();
  `);

  const progressText = await waitFor(async () => {
    const text = await evaluate("document.getElementById('progressText')?.textContent || ''");
    return text.includes("Trial 1/3") && text.includes("saved 0") ? text : "";
  });
  assert.match(progressText, /Trial 1\/3/);
  assert.match(progressText, /saved 0/);

  const initialComparisonMarkButton = await evaluate(`(() => {
    const button = document.getElementById("comparisonMarkBtn");
    return {
      disabled: Boolean(button?.disabled),
      text: button?.textContent.trim() || "",
      pressed: button?.getAttribute("aria-pressed") || "",
      pairId: document.getElementById("comparisonWorkspace")?.dataset.pairId || "",
    };
  })()`);
  assert.equal(initialComparisonMarkButton.disabled, false);
  assert.equal(initialComparisonMarkButton.text, "Mark Comparison");
  assert.equal(initialComparisonMarkButton.pressed, "false");
  assert.ok(initialComparisonMarkButton.pairId);

  await evaluate(`document.getElementById("comparisonMarkBtn").click()`);
  const markedComparison = await waitFor(async () => {
    const value = await evaluate(`(() => {
      const button = document.getElementById("comparisonMarkBtn");
      const text = new TextDecoder().decode(window.__mosFakeDirectory.files.get("marked_comparisons.csv"));
      const rows = window.MosGroupCsv.parseCsv(text);
      const row = rows[0] || {};
      return {
        buttonText: button?.textContent.trim() || "",
        pressed: button?.getAttribute("aria-pressed") || "",
        rowCount: rows.length,
        markOrder: row.mark_order || "",
        pairId: row.pair_id || "",
        expert: row.expert_id || "",
        group: row.group_id || "",
        paths: [row.input_path, row.candidate_a_path, row.candidate_b_path],
      };
    })()`);
    return value.buttonText === "Unmark Comparison" && value.rowCount === 1 ? value : null;
  });
  assert.equal(markedComparison.pressed, "true");
  assert.equal(markedComparison.markOrder, "1");
  assert.equal(markedComparison.pairId, initialComparisonMarkButton.pairId);
  assert.equal(markedComparison.expert, "smoke");
  assert.equal(markedComparison.group, "A");
  assert.equal(markedComparison.paths.every((imagePath) => imagePath.includes("/images/scenes/")), true);

  await evaluate(`
    document.getElementById("pairMode").value = "marked";
    document.getElementById("startBtn").click();
  `);
  const restoredMarkedComparison = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      progress: document.getElementById("progressText")?.textContent || "",
      pairId: document.getElementById("comparisonWorkspace")?.dataset.pairId || "",
      buttonText: document.getElementById("comparisonMarkBtn")?.textContent.trim() || "",
      paths: ["inputImage", "aImage", "bImage"].map((id) =>
        document.getElementById(id)?.getAttribute("src") || ""),
    }))()`);
    return value.progress.includes("Trial 1/1") ? value : null;
  });
  assert.equal(restoredMarkedComparison.pairId, initialComparisonMarkButton.pairId);
  assert.equal(restoredMarkedComparison.buttonText, "Unmark Comparison");
  assert.deepEqual(restoredMarkedComparison.paths, markedComparison.paths);

  await evaluate(`document.getElementById("comparisonMarkBtn").click()`);
  const unmarkedComparison = await waitFor(async () => {
    const value = await evaluate(`(() => {
      const button = document.getElementById("comparisonMarkBtn");
      const text = new TextDecoder().decode(window.__mosFakeDirectory.files.get("marked_comparisons.csv"));
      return {
        buttonText: button?.textContent.trim() || "",
        pressed: button?.getAttribute("aria-pressed") || "",
        rowCount: window.MosGroupCsv.parseCsv(text).length,
      };
    })()`);
    return value.buttonText === "Mark Comparison" && value.rowCount === 0 ? value : null;
  });
  assert.equal(unmarkedComparison.pressed, "false");

  await evaluate(`
    document.getElementById("pairMode").value = "compact";
    document.getElementById("startBtn").click();
  `);
  await waitFor(async () => {
    const text = await evaluate(`document.getElementById("progressText")?.textContent || ""`);
    return text.includes("Trial 1/3") ? text : "";
  });

  const migratedStorage = await evaluate(`(() => {
    const newKey = "eciqa-pref:smoke:A:all:compact:3:scene-order";
    const payload = JSON.parse(localStorage.getItem(newKey) || "null");
    return {
      newKey,
      newExists: Boolean(payload),
      legacyExists: localStorage.getItem(${JSON.stringify(legacySetup.legacyKey)}) !== null,
      rootGroup: payload?.group_id || "",
      resultGroups: (payload?.results || []).map((result) => result.group_id),
      resultExperts: (payload?.results || []).map((result) => result.expert_id),
    };
  })()`);
  assert.deepEqual(migratedStorage, {
    newKey: "eciqa-pref:smoke:A:all:compact:3:scene-order",
    newExists: true,
    legacyExists: false,
    rootGroup: "A",
    resultGroups: ["A"],
    resultExperts: ["smoke"],
  });

  const imageStatus = await waitFor(async () => {
    const status = await evaluate(`
      Promise.all(["inputImage", "aImage", "bImage"].map(async (id) => {
        const img = document.getElementById(id);
        if (img.decode) {
          try { await img.decode(); } catch {}
        }
        return {
          id,
          src: img.currentSrc || img.src,
          complete: img.complete,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight
        };
      }))
    `);
    return status.every((img) => img.src.includes("/scenes/") && img.complete && img.naturalWidth > 0)
      ? status
      : null;
  }, 20000);
  assert.equal(imageStatus.length, 3);

  const landscapeLayout = await layoutSnapshot();
  assertJudgementOverflow(landscapeLayout, "annotation landscape");
  assert.ok(landscapeLayout, "comparisonWorkspace must exist");
  assert.match(landscapeLayout.className, /layout-landscape/);
  assert.equal(landscapeLayout.chromeInsideJudgement, true);
  assert.equal(landscapeLayout.resolutionLabelCount, 0);
  assert.ok(landscapeLayout.workspace.top <= 16, "workspace should start at the top of the app");
  assert.deepEqual(landscapeLayout.labels, ["Input", "Candidate A", "Candidate B"]);
  assert.ok(landscapeLayout.judgement.right <= landscapeLayout.input.left + 1);
  assert.ok(landscapeLayout.judgement.bottom <= landscapeLayout.a.top + 1);
  assert.ok(landscapeLayout.a.right <= landscapeLayout.b.left + 1);
  assert.equal(overlaps(landscapeLayout.judgement, landscapeLayout.input), false);
  assert.equal(overlaps(landscapeLayout.judgement, landscapeLayout.a), false);
  assert.equal(overlaps(landscapeLayout.input, landscapeLayout.b), false);
  Object.entries(landscapeLayout.frames).forEach(([label, frame]) => assertAdaptiveFrame(frame, label));
  await captureScreenshot("landscape-desktop.png");

  const portraitSeed = await evaluate(`(() => {
    const allScenes = window.MosCore.buildScenes(window.MOS_RECORDS);
    for (let index = 0; index < 50; index += 1) {
      const seed = "portrait-" + index;
      const first = window.MosCore.buildTrials(allScenes, {
        dataset: "all",
        pairMode: "compact",
        maxTrials: 1,
        seed,
      })[0];
      if (first && Number(first.input.height) > Number(first.input.width)) return seed;
    }
    return "";
  })()`);
  assert.ok(portraitSeed, "a deterministic portrait-first seed should be found");
  await evaluate(`
    document.getElementById("seedText").value = ${JSON.stringify(portraitSeed)};
    document.getElementById("startBtn").click();
  `);
  await waitFor(async () => {
    const ready = await evaluate(`Promise.all(["inputImage", "aImage", "bImage"].map(async (id) => {
      const image = document.getElementById(id);
      if (image.decode) { try { await image.decode(); } catch {} }
      return image.complete && image.naturalWidth > 0;
    })).then((states) => states.every(Boolean) && document.getElementById("comparisonWorkspace")?.classList.contains("layout-portrait"))`);
    return ready;
  }, 20000);

  const portraitLayout = await layoutSnapshot();
  assertJudgementOverflow(portraitLayout, "annotation portrait");
  assert.match(portraitLayout.className, /layout-portrait/);
  assert.ok(portraitLayout.judgement.bottom <= portraitLayout.input.top + 1);
  assert.ok(portraitLayout.input.right <= portraitLayout.a.left + 1);
  assert.ok(portraitLayout.a.right <= portraitLayout.b.left + 1);
  assert.equal(overlaps(portraitLayout.judgement, portraitLayout.input), false);
  assert.equal(overlaps(portraitLayout.input, portraitLayout.a), false);
  assert.equal(overlaps(portraitLayout.a, portraitLayout.b), false);
  Object.entries(portraitLayout.frames).forEach(([label, frame]) => assertAdaptiveFrame(frame, label));
  assert.ok(
    portraitLayout.inputImage.left >= portraitLayout.inputBox.left - 1 &&
      portraitLayout.inputImage.right <= portraitLayout.inputBox.right + 1 &&
      portraitLayout.inputImage.top >= portraitLayout.inputBox.top - 1 &&
      portraitLayout.inputImage.bottom <= portraitLayout.inputBox.bottom + 1,
    `portrait image must be contained: ${JSON.stringify({
      box: portraitLayout.inputBox,
      image: portraitLayout.inputImage,
    })}`,
  );

  const lensState = await evaluate(`(() => {
    const image = document.getElementById("inputImage");
    const bounds = image.getBoundingClientRect();
    document.getElementById("comparisonWorkspace").dispatchEvent(new MouseEvent("mousemove", {
      bubbles: true,
      clientX: bounds.left + bounds.width / 2,
      clientY: bounds.top + bounds.height / 2,
    }));
    return Array.from(document.querySelectorAll(".lens"), (lens) => ({
      active: lens.classList.contains("active"),
      backgroundImage: lens.style.backgroundImage,
      backgroundPosition: lens.style.backgroundPosition,
    }));
  })()`);
  assert.equal(lensState.length, 3);
  assert.equal(lensState.every((lens) => lens.active), true);
  assert.equal(lensState.every((lens) => lens.backgroundImage.includes("scenes")), true);
  assert.equal(lensState.every((lens) => lens.backgroundPosition.length > 0), true);
  await evaluate(`
    document.getElementById("comparisonWorkspace").dispatchEvent(new MouseEvent("mouseleave"));
  `);
  assert.equal(
    await evaluate(`Array.from(document.querySelectorAll(".lens"), (lens) => lens.classList.contains("active")).every((active) => !active)`),
    true,
  );
  await captureScreenshot("portrait-desktop.png");

  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 720,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await sleep(300);
  const narrowLayout = await layoutSnapshot();
  assertJudgementOverflow(narrowLayout, "annotation narrow");
  assert.match(narrowLayout.className, /layout-portrait/);
  assert.ok(narrowLayout.judgement.bottom <= narrowLayout.input.top + 1);
  assert.ok(narrowLayout.input.bottom <= narrowLayout.a.top + 1);
  assert.ok(narrowLayout.a.bottom <= narrowLayout.b.top + 1);
  assert.equal(overlaps(narrowLayout.judgement, narrowLayout.input), false);
  assert.equal(overlaps(narrowLayout.input, narrowLayout.a), false);
  assert.equal(overlaps(narrowLayout.a, narrowLayout.b), false);
  await captureScreenshot("portrait-narrow.png");
  await cdp.send("Emulation.clearDeviceMetricsOverride");
  await sleep(300);

  const pairBeforeSave = await evaluate(
    `document.getElementById("comparisonWorkspace")?.dataset.pairId || ""`,
  );
  assert.ok(pairBeforeSave, "the displayed workspace must expose the canonical pair id");

  await evaluate(`
    document.querySelector('.choiceRow[data-key="input_vs_a"] button[data-value="left"]').click();
    document.querySelector('.choiceRow[data-key="input_vs_b"] button[data-value="right"]').click();
    document.querySelector('.choiceRow[data-key="a_vs_b"] button[data-value="left"]').click();
  `);
  assert.match(await evaluate("document.getElementById('rankingText').textContent"), /Conflict/);
  assert.equal(await evaluate("document.getElementById('nextBtn').disabled"), true,
    "cyclic preferences must not be saved");

  await evaluate(`
    document.querySelector('.choiceRow[data-key="input_vs_a"] button[data-value="left"]').click();
    document.querySelector('.choiceRow[data-key="input_vs_b"] button[data-value="left"]').click();
    document.querySelector('.choiceRow[data-key="a_vs_b"] button[data-value="similar"]').click();
  `);
  const rankingText = await evaluate("document.getElementById('rankingText').textContent");
  const nextDisabled = await evaluate("document.getElementById('nextBtn').disabled");
  assert.equal(rankingText, "Input>A=B");
  assert.equal(nextDisabled, false);

  await evaluate("document.getElementById('nextBtn').click()");
  const savedProgress = await waitFor(async () => {
    const text = await evaluate("document.getElementById('progressText')?.textContent || ''");
    return text.includes("saved 1") ? text : "";
  });
  assert.match(savedProgress, /saved 1/);

  const groupCsvState = await evaluate(`(() => {
    const text = new TextDecoder().decode(window.__mosFakeDirectory.files.get("group_A.csv"));
    const rows = window.MosGroupCsv.parseCsv(text);
    return {
      count: rows.length,
      pairIds: rows.map((row) => row.pair_id),
      groups: rows.map((row) => row.group_id),
      experts: rows.map((row) => row.expert_id),
    };
  })()`);
  assert.equal(groupCsvState.count, 2, "legacy and newly saved results must both be in Group A CSV");
  assert.equal(groupCsvState.pairIds.includes(pairBeforeSave), true);
  assert.equal(
    groupCsvState.pairIds.some((pairId) => pairId.includes("removed_smoke_scene")),
    false,
    "local records for removed scenes must not be merged back into Group A CSV",
  );
  assert.deepEqual(groupCsvState.groups, ["A", "A"]);
  assert.deepEqual(groupCsvState.experts, ["smoke", "smoke"]);

  const exportNames = await evaluate(`(() => {
    const names = [];
    const originalClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function captureDownload() {
      names.push(this.download);
    };
    document.getElementById("exportCsvBtn").click();
    document.getElementById("exportJsonBtn").click();
    HTMLAnchorElement.prototype.click = originalClick;
    return names;
  })()`);
  assert.deepEqual(exportNames, [
    "smoke_group_A_mos_results.csv",
    "smoke_group_A_mos_results.json",
  ]);

  await evaluate(`
    document.getElementById("expertId").value = "smoke-next";
    document.getElementById("startBtn").click();
  `);
  const handoffPair = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      progress: document.getElementById("progressText")?.textContent || "",
      pairId: document.getElementById("comparisonWorkspace")?.dataset.pairId || "",
    }))()`);
    return value.progress.includes("Trial 1/3") && value.pairId && value.pairId !== pairBeforeSave
      ? value
      : null;
  });
  assert.match(handoffPair.progress, /saved 0/);
  assert.equal(groupCsvState.pairIds.includes(handoffPair.pairId), false);

  const coverageSetup = await evaluate(`(() => {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const allScenes = window.MosCore.buildScenes(window.MOS_RECORDS);
    const baseTrials = window.MosCore.buildTrials(allScenes, {
      dataset: "MSEC",
      pairMode: window.MosCore.COLOR_FIX_COVERAGE_MODE,
      seed: "scene-order",
    });
    const groupARows = window.MosGroupCsv.parseCsv(
      decoder.decode(window.__mosFakeDirectory.files.get("group_A.csv")),
    );
    const seenInA = new Set(groupARows.map((row) => row.scene_id));
    const groupBOnlyTrial = baseTrials.find((trial) => !seenInA.has(trial.scene_id));
    if (!groupBOnlyTrial) return null;
    const groupBResult = {
      expert_id: "coverage-b",
      group_id: "B",
      pair_id: groupBOnlyTrial.pair_id,
      trial_id: groupBOnlyTrial.trial_id,
      dataset: groupBOnlyTrial.dataset,
      scene_id: groupBOnlyTrial.scene_id,
      input_method: groupBOnlyTrial.input.method,
      candidate_a_method: groupBOnlyTrial.a.method,
      candidate_b_method: groupBOnlyTrial.b.method,
      input_path: groupBOnlyTrial.input.path,
      candidate_a_path: groupBOnlyTrial.a.path,
      candidate_b_path: groupBOnlyTrial.b.path,
      input_vs_a: "left",
      input_vs_b: "left",
      a_vs_b: "similar",
      ranking: "Input>A=B",
      elapsed_ms: 1000,
      timestamp: "2026-08-03T00:00:00.000Z",
    };
    window.__mosFakeDirectory.files.set(
      "group_B.csv",
      encoder.encode(window.MosGroupCsv.serializeResults([groupBResult])),
    );
    const globallySeen = new Set([...seenInA, groupBOnlyTrial.scene_id]);
    const expectedTrial = baseTrials.find((trial) => !globallySeen.has(trial.scene_id));
    return {
      groupBOnlyScene: groupBOnlyTrial.scene_id,
      expectedScene: expectedTrial?.scene_id || "",
    };
  })()`);
  assert.ok(coverageSetup?.expectedScene);

  await evaluate(`
    document.getElementById("expertId").value = "coverage-a";
    document.getElementById("groupId").value = "A";
    document.getElementById("datasetFilter").value = "MSEC";
    document.getElementById("pairMode").value = "unseen";
    document.getElementById("maxTrials").value = "2";
    document.getElementById("seedText").value = "scene-order";
    document.getElementById("startBtn").click();
  `);
  const coverageState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      progress: document.getElementById("progressText")?.textContent || "",
      sceneText: document.getElementById("sceneText")?.textContent || "",
      pairId: document.getElementById("comparisonWorkspace")?.dataset.pairId || "",
      candidatePaths: ["aImage", "bImage"].map((id) =>
        document.getElementById(id)?.getAttribute("src") || ""),
    }))()`);
    return value.progress.includes("Trial 1/2") ? value : null;
  });
  assert.equal(coverageState.sceneText, `MSEC | ${coverageSetup.expectedScene}`);
  assert.notEqual(coverageSetup.expectedScene, coverageSetup.groupBOnlyScene);
  assert.match(coverageState.pairId, /cotf_mefssim_0\.5x_color_fix/);
  assert.equal(
    coverageState.candidatePaths.some((imagePath) =>
      imagePath.includes("cotf_mefssim_0p5x_color_fix")),
    true,
  );

  // Reconnect the same in-memory folder after an actual document reload.
  const folderFiles = await evaluate(`Array.from(window.__mosFakeDirectory.files,
    ([name, bytes]) => [name, Array.from(bytes)])`);
  await cdp.send("Page.reload");
  await waitFor(async () => await evaluate(`document.readyState === "complete" &&
    document.getElementById("metaText")?.textContent.includes("5200 images")`));
  await evaluate(`window.__smokeFolderFiles = ${JSON.stringify(folderFiles)}`);
  await evaluate(folderFixture);
  await waitFor(async () => await evaluate(`
    document.getElementById("workingFolderStatus")?.textContent.includes("handoff-folder") &&
    !document.getElementById("startBtn")?.disabled`));
  assert.deepEqual(await evaluate(`Array.from(window.__mosFakeDirectory.files,
    ([name, bytes]) => [name, Array.from(bytes)])`), folderFiles,
  "reconnecting after reload must preserve saved CSV contents");
  await evaluate(`
    document.getElementById("expertId").value = "reload-smoke";
    document.getElementById("groupId").value = "A";
    document.getElementById("pairMode").value = "compact";
    document.getElementById("maxTrials").value = "3";
    document.getElementById("seedText").value = "scene-order";
    document.getElementById("startBtn").click();
  `);
  const reloadedPair = await waitFor(async () => await evaluate(`
    document.getElementById("comparisonWorkspace")?.dataset.pairId || ""`));
  assert.equal(groupCsvState.pairIds.includes(reloadedPair), false,
    "reloaded sessions must skip comparisons already saved for the group");

  await cdp.send("Page.navigate", { url: reviewUrl });
  await waitFor(async () => {
    const ready = await evaluate(`(() => ({
      readyState: document.readyState,
      title: document.title,
      hasReviewCore: Boolean(window.MosGroupCsv?.annotationReviewId),
      metaText: document.getElementById("metaText")?.textContent || ""
    }))()`);
    return ready.readyState === "complete" && ready.title.includes("Review") &&
      ready.hasReviewCore && ready.metaText.includes("images") ? ready : null;
  }, 30000);

  const reviewControls = await evaluate(`(() => ({
    reviewerLabel: document.querySelector('label[for="reviewerId"]')?.childNodes[0]?.textContent.trim() || "",
    reviewGroupOptions: Array.from(document.getElementById("reviewGroupId")?.options || [], (option) => option.value),
    sourceGroupOptions: Array.from(document.getElementById("sourceGroupFilter")?.options || [], (option) => option.value),
    statusOptions: Array.from(document.getElementById("reviewStatusFilter")?.options || [], (option) => option.value),
    reviewedText: document.getElementById("reviewedBtn")?.textContent.trim() || "",
    skipText: document.getElementById("skipBtn")?.textContent.trim() || "",
    editableChoiceCount: document.querySelectorAll(".choiceRow button:not(:disabled)").length
  }))()`);
  assert.deepEqual(reviewControls, {
    reviewerLabel: "Reviewer ID",
    reviewGroupOptions: ["A", "B", "C", "D", "E"],
    sourceGroupOptions: ["all", "A", "B", "C", "D", "E"],
    statusOptions: ["pending", "all", "reviewed"],
    reviewedText: "Reviewed",
    skipText: "Skip",
    editableChoiceCount: 9,
  });

  await evaluate(`(() => {
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const scenes = window.MosCore.buildScenes(window.MOS_RECORDS);
    const trials = window.MosCore.buildTrials(scenes, {
      dataset: "all",
      pairMode: "compact",
      maxTrials: 2,
      seed: "scene-order",
    });
    const sourceResults = trials.map((trial, index) => {
      const choices = index === 0
        ? { input_vs_a: "left", input_vs_b: "left", a_vs_b: "similar" }
        : { input_vs_a: "left", input_vs_b: "left", a_vs_b: "left" };
      return {
        expert_id: "source-" + (index + 1),
        group_id: "A",
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
        input_vs_a: choices.input_vs_a,
        input_vs_b: choices.input_vs_b,
        a_vs_b: choices.a_vs_b,
        ranking: window.MosCore.summarizeRanking(choices).text,
        elapsed_ms: 1000 + index,
        timestamp: "2026-08-06T00:0" + index + ":00.000Z",
      };
    });
    const files = new Map([
      ["group_A.csv", encoder.encode(window.MosGroupCsv.serializeResults(sourceResults))],
    ]);
    const directory = {
      kind: "directory",
      name: "review-folder",
      files,
      async queryPermission() { return "granted"; },
      async requestPermission() { return "granted"; },
      async getFileHandle(name, options = {}) {
        if (!files.has(name)) {
          if (!options.create) throw new DOMException("Missing file", "NotFoundError");
          files.set(name, new Uint8Array());
        }
        return {
          kind: "file",
          name,
          async getFile() {
            const snapshot = files.get(name).slice();
            return {
              size: snapshot.byteLength,
              async text() { return decoder.decode(snapshot); },
            };
          },
          async createWritable(options = {}) {
            let content = options.keepExistingData ? files.get(name).slice() : new Uint8Array();
            let position = 0;
            const ensureSize = (size) => {
              if (size <= content.byteLength) return;
              const expanded = new Uint8Array(size);
              expanded.set(content);
              content = expanded;
            };
            return {
              async seek(nextPosition) { position = Number(nextPosition); },
              async write(value) {
                const incoming = value instanceof Uint8Array ? value : encoder.encode(String(value));
                ensureSize(position + incoming.byteLength);
                content.set(incoming, position);
                position += incoming.byteLength;
              },
              async close() { files.set(name, content); },
              async abort() {},
            };
          },
        };
      },
    };
    window.__reviewFakeDirectory = directory;
    window.showDirectoryPicker = async () => directory;
    document.getElementById("workingFolderBtn").click();
  })()`);

  const reviewFolder = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      statusText: document.getElementById("workingFolderStatus")?.textContent || "",
      filenames: Array.from(window.__reviewFakeDirectory?.files?.keys() || []).sort(),
      startDisabled: Boolean(document.getElementById("startBtn")?.disabled)
    }))()`);
    return value.statusText.includes("review-folder") && value.filenames.length === 11 ? value : null;
  });
  assert.equal(reviewFolder.startDisabled, false);
  assert.deepEqual(
    reviewFolder.filenames.filter((name) => name.endsWith("_reviewed.csv")),
    [
      "group_A_reviewed.csv",
      "group_B_reviewed.csv",
      "group_C_reviewed.csv",
      "group_D_reviewed.csv",
      "group_E_reviewed.csv",
    ],
  );

  await evaluate(`
    document.getElementById("reviewerId").value = "review-smoke";
    document.getElementById("reviewGroupId").value = "A";
    document.getElementById("sourceGroupFilter").value = "all";
    document.getElementById("datasetFilter").value = "all";
    document.getElementById("reviewStatusFilter").value = "pending";
    document.getElementById("maxReviews").value = "2";
    document.getElementById("startBtn").click();
  `);
  const firstReviewItem = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      progress: document.getElementById("progressText")?.textContent || "",
      sceneText: document.getElementById("sceneText")?.textContent || "",
      pairId: document.getElementById("comparisonWorkspace")?.dataset.pairId || "",
      annotationId: document.getElementById("comparisonWorkspace")?.dataset.annotationId || "",
      selectedChoices: document.querySelectorAll(".choiceRow button.selected").length,
      reviewed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
      marked: document.getElementById("comparisonMarkBtn")?.getAttribute("aria-pressed")
    }))()`);
    return value.progress.includes("Review 1/2") && value.annotationId ? value : null;
  });
  assert.match(firstReviewItem.sceneText, /Source Group A \| Expert source-1/);
  assert.equal(firstReviewItem.selectedChoices, 3);
  assert.equal(firstReviewItem.reviewed, "false");
  assert.equal(firstReviewItem.marked, "false");

  await waitFor(async () => {
    const ready = await evaluate(`Promise.all(["inputImage", "aImage", "bImage"].map(async (id) => {
      const image = document.getElementById(id);
      if (image.decode) { try { await image.decode(); } catch {} }
      return image.complete && image.naturalWidth > 0;
    })).then((states) => states.every(Boolean))`);
    return ready;
  }, 20000);
  const reviewLayout = await layoutSnapshot();
  assertJudgementOverflow(reviewLayout, "review landscape");
  assert.equal(reviewLayout.chromeInsideJudgement, true);
  assert.deepEqual(reviewLayout.labels, ["Input", "Candidate A", "Candidate B"]);
  Object.entries(reviewLayout.frames).forEach(([label, frame]) => assertAdaptiveFrame(frame, label));

  await evaluate(`document.getElementById("comparisonMarkBtn").click()`);
  const sharedMarkState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      pressed: document.getElementById("comparisonMarkBtn")?.getAttribute("aria-pressed"),
      rows: window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("marked_comparisons.csv"))).length
    }))()`);
    return value.pressed === "true" && value.rows === 1 ? value : null;
  });
  assert.deepEqual(sharedMarkState, { pressed: "true", rows: 1 });

  await evaluate(`
    document.querySelector('.choiceRow[data-key="input_vs_b"] button[data-value="right"]').click();
    document.querySelector('.choiceRow[data-key="a_vs_b"] button[data-value="left"]').click();
  `);
  const conflictReviewState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      ranking: document.getElementById("rankingText")?.textContent || "",
      reviewedDisabled: Boolean(document.getElementById("reviewedBtn")?.disabled)
    }))()`);
    return value.ranking.includes("Conflict") ? value : null;
  });
  assert.equal(conflictReviewState.reviewedDisabled, true);

  await evaluate(`document.querySelector('.choiceRow[data-key="input_vs_b"] button[data-value="left"]').click()`);
  const editedChoiceState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      ranking: document.getElementById("rankingText")?.textContent || "",
      selected: document.querySelector('.choiceRow[data-key="a_vs_b"] button[data-value="left"]')?.classList.contains("selected"),
      reviewedDisabled: Boolean(document.getElementById("reviewedBtn")?.disabled)
    }))()`);
    return value.ranking === "Input>A>B" && value.selected ? value : null;
  });
  assert.equal(editedChoiceState.reviewedDisabled, false);

  await evaluate(`document.getElementById("reviewedBtn").click()`);
  const persistedReview = await waitFor(async () => {
    const value = await evaluate(`(() => {
      const rows = window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("group_A_reviewed.csv")));
      return {
        pressed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
        highlighted: document.getElementById("reviewedBtn")?.classList.contains("reviewed"),
        rows: rows.length,
        reviewerId: rows[0]?.reviewer_id || "",
        sourceGroupId: rows[0]?.source_group_id || "",
        annotationId: rows[0]?.annotation_id || "",
        aVsB: rows[0]?.a_vs_b || "",
        ranking: rows[0]?.ranking || "",
        sourceAVsB: window.MosGroupCsv.parseCsv(new TextDecoder().decode(
          window.__reviewFakeDirectory.files.get("group_A.csv")))[0]?.a_vs_b || ""
      };
    })()`);
    return value.pressed === "true" && value.rows === 1 ? value : null;
  });
  assert.equal(persistedReview.highlighted, true);
  assert.equal(persistedReview.reviewerId, "review-smoke");
  assert.equal(persistedReview.sourceGroupId, "A");
  assert.equal(persistedReview.annotationId, firstReviewItem.annotationId);
  assert.equal(persistedReview.aVsB, "left");
  assert.equal(persistedReview.ranking, "Input>A>B");
  assert.equal(persistedReview.sourceAVsB, "similar", "review edits must not overwrite the source CSV");

  await evaluate(`document.getElementById("skipBtn").click()`);
  const skippedItem = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      progress: document.getElementById("progressText")?.textContent || "",
      annotationId: document.getElementById("comparisonWorkspace")?.dataset.annotationId || "",
      rows: window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("group_A_reviewed.csv"))).length
    }))()`);
    return value.progress.includes("Review 2/2") ? value : null;
  });
  assert.notEqual(skippedItem.annotationId, firstReviewItem.annotationId);
  assert.equal(skippedItem.rows, 1, "Skip must not write a review row");

  await evaluate(`document.getElementById("prevBtn").click()`);
  const restoredReviewState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      annotationId: document.getElementById("comparisonWorkspace")?.dataset.annotationId || "",
      reviewed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
      marked: document.getElementById("comparisonMarkBtn")?.getAttribute("aria-pressed"),
      aVsB: document.querySelector('.choiceRow[data-key="a_vs_b"] button.selected')?.dataset.value || "",
      ranking: document.getElementById("rankingText")?.textContent || ""
    }))()`);
    return value.annotationId === firstReviewItem.annotationId ? value : null;
  });
  assert.equal(restoredReviewState.annotationId, firstReviewItem.annotationId);
  assert.equal(restoredReviewState.reviewed, "true");
  assert.equal(restoredReviewState.marked, "true");
  assert.equal(restoredReviewState.aVsB, "left");
  assert.equal(restoredReviewState.ranking, "Input>A>B");
  await captureScreenshot("review-desktop.png");

  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 720,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await sleep(300);
  const reviewNarrowLayout = await layoutSnapshot();
  assertJudgementOverflow(reviewNarrowLayout, "review narrow");
  assert.ok(reviewNarrowLayout.judgement.bottom <= reviewNarrowLayout.input.top + 1);
  assert.ok(reviewNarrowLayout.input.bottom <= reviewNarrowLayout.a.top + 1);
  assert.ok(reviewNarrowLayout.a.bottom <= reviewNarrowLayout.b.top + 1);
  assert.equal(overlaps(reviewNarrowLayout.judgement, reviewNarrowLayout.input), false);
  assert.equal(overlaps(reviewNarrowLayout.input, reviewNarrowLayout.a), false);
  assert.equal(overlaps(reviewNarrowLayout.a, reviewNarrowLayout.b), false);
  await captureScreenshot("review-narrow.png");
  await cdp.send("Emulation.clearDeviceMetricsOverride");
  await sleep(300);

  await evaluate(`document.querySelector('.choiceRow[data-key="a_vs_b"] button[data-value="similar"]').click()`);
  const dirtyReviewState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      pressed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
      highlighted: document.getElementById("reviewedBtn")?.classList.contains("reviewed"),
      ranking: document.getElementById("rankingText")?.textContent || "",
      savedAVsB: window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("group_A_reviewed.csv")))[0]?.a_vs_b || ""
    }))()`);
    return value.pressed === "false" && value.ranking === "Input>A=B" ? value : null;
  });
  assert.equal(dirtyReviewState.highlighted, false);
  assert.equal(dirtyReviewState.savedAVsB, "left", "editing must not persist before Reviewed is clicked");

  await evaluate(`document.getElementById("reviewedBtn").click()`);
  const updatedReviewState = await waitFor(async () => {
    const value = await evaluate(`(() => {
      const rows = window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("group_A_reviewed.csv")));
      return {
        pressed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
        rows: rows.length,
        aVsB: rows[0]?.a_vs_b || "",
        ranking: rows[0]?.ranking || ""
      };
    })()`);
    return value.pressed === "true" && value.aVsB === "similar" ? value : null;
  });
  assert.deepEqual(updatedReviewState, {
    pressed: "true",
    rows: 1,
    aVsB: "similar",
    ranking: "Input>A=B",
  });

  await evaluate(`document.getElementById("reviewedBtn").click()`);
  const removedReviewState = await waitFor(async () => {
    const value = await evaluate(`(() => ({
      pressed: document.getElementById("reviewedBtn")?.getAttribute("aria-pressed"),
      rows: window.MosGroupCsv.parseCsv(new TextDecoder().decode(
        window.__reviewFakeDirectory.files.get("group_A_reviewed.csv"))).length
    }))()`);
    return value.pressed === "false" && value.rows === 0 ? value : null;
  });
  assert.deepEqual(removedReviewState, { pressed: "false", rows: 0 });

  const browserErrors = cdp.events.filter((event) =>
    event.method === "Runtime.exceptionThrown" ||
    (event.method === "Log.entryAdded" && event.level === "error"));
  assert.deepEqual(browserErrors, [], `browser errors: ${JSON.stringify(browserErrors)}`);

  const report = {
    status: "passed",
    checked_at: new Date().toISOString(),
    image_count: 5200,
    source: appUrl,
    folder_access: "in-memory File System Access test double",
    native_folder_writes_verified: false,
    screenshots: screenshotDir,
    desktop_controls: {
      annotation: landscapeLayout.judgementOverflow,
      review: reviewLayout.judgementOverflow,
    },
    checks: [
      "real image decoding and adaptive landscape/portrait/narrow layout",
      "three-way comparison conflict rejection",
      "group CSV save and CSV/JSON export",
      "saved comparisons skipped during group handoff",
      "folder reconnection after document reload",
      "marked comparisons and unseen-scene coverage",
      "review edits persisted without overwriting source CSV",
      "review restore, update, skip, and undo",
      "no browser runtime or console errors",
    ],
  };
  fs.writeFileSync(path.join(screenshotDir, "browser_smoke.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
} finally {
  if (cdp) cdp.socket.close();
  if (process.platform === "win32" && browser.pid) {
    try {
      await execFileAsync("taskkill.exe", ["/PID", String(browser.pid), "/T", "/F"], { windowsHide: true });
    } catch {
      browser.kill();
    }
  } else {
    browser.kill();
  }
  await new Promise((resolve) => {
    const timeout = setTimeout(resolve, 3000);
    browser.once("exit", () => {
      clearTimeout(timeout);
      resolve();
    });
  });
  try {
    fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 8, retryDelay: 250 });
  } catch (error) {
    console.warn(`Warning: could not remove temporary browser profile: ${error.message}`);
  }
}
