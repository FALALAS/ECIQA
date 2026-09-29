(function attachMosCore(root) {
const LABELS = ["Input", "A", "B"];
const SCENE_ORDER_SEED = "scene-order";
const COLOR_FIX_COVERAGE_MODE = "unseen";
const COLOR_FIX_METHOD = "cotf_mefssim_0.5x_color_fix";

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function canonicalSeed(value) {
  const seed = String(value || "sec-mos").trim() || "sec-mos";
  return seed.toLowerCase() === SCENE_ORDER_SEED ? SCENE_ORDER_SEED : seed;
}

function canonicalPairId(dataset, sceneId, firstMethod, secondMethod) {
  const methods = [String(firstMethod || ""), String(secondMethod || "")].sort();
  return [dataset, sceneId, ...methods].map((value) => String(value || "")).join("::");
}

function imageOrientation(record) {
  const width = numberValue(record?.width);
  const height = numberValue(record?.height);
  return width > 0 && height > width ? "portrait" : "landscape";
}

function hashString(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createRng(seedText) {
  let state = hashString(seedText || "sec-mos");
  return function rng() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, rng) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function sortedCandidates(candidates) {
  return candidates.slice().sort((a, b) => {
    const orderDiff = numberValue(a.display_order) - numberValue(b.display_order);
    if (orderDiff !== 0) return orderDiff;
    return String(a.method).localeCompare(String(b.method));
  });
}

function buildScenes(records) {
  const byScene = new Map();
  for (const record of records || []) {
    if (!record || !record.scene_id) continue;
    if (!byScene.has(record.scene_id)) {
      byScene.set(record.scene_id, {
        dataset: record.dataset || "",
        scene_id: record.scene_id,
        sample_id: record.sample_id || record.scene_id,
        manifest_order: byScene.size,
        input: null,
        candidates: [],
      });
    }
    const scene = byScene.get(record.scene_id);
    if (record.item_type === "input") {
      scene.input = record;
    } else if (record.item_type === "candidate") {
      scene.candidates.push(record);
    }
  }

  return Array.from(byScene.values())
    .map((scene) => ({ ...scene, candidates: sortedCandidates(scene.candidates) }))
    .filter((scene) => scene.input && scene.candidates.length >= 2)
    .sort((a, b) => a.scene_id.localeCompare(b.scene_id));
}

function candidatePairs(candidates, pairMode) {
  const pairs = [];
  if (pairMode === "compact") {
    const byMethod = new Map(candidates.map((candidate) => [candidate.method, candidate]));
    const wanted = [
      ["cotf_original", "cotf_mefssim_0.5x_color_fix"],
      ["fecnet_original", "fecnet_mefssim_0.5x"],
      ["cotf_original", "fecnet_original"],
      ["cotf_mefssim_0.5x_color_fix", "fecnet_mefssim_0.5x"],
      ["cotf_original", "cotf_fusegt"],
      ["fecnet_original", "fecnet_fusegt"],
      ["cotf_fusegt", "fecnet_fusegt"],
    ];
    for (const [left, right] of wanted) {
      if (byMethod.has(left) && byMethod.has(right)) {
        pairs.push([byMethod.get(left), byMethod.get(right)]);
      }
    }
    if (pairs.length > 0) return pairs;
  }

  for (let i = 0; i < candidates.length; i += 1) {
    for (let j = i + 1; j < candidates.length; j += 1) {
      pairs.push([candidates[i], candidates[j]]);
    }
  }
  return pairs;
}

function colorFixCoveragePairs(scene, seed) {
  const colorFix = scene.candidates.find(({ method }) => method === COLOR_FIX_METHOD);
  const opponents = scene.candidates.filter(({ method }) => method !== COLOR_FIX_METHOD);
  if (!colorFix || opponents.length === 0) return [];

  const rng = createRng(`${seed}:${scene.scene_id}:${COLOR_FIX_COVERAGE_MODE}`);
  const opponent = opponents[Math.floor(rng() * opponents.length)];
  return [[colorFix, opponent]];
}

function prioritizeUnseenSceneTrials(trials, seenSceneIds) {
  const seen = seenSceneIds instanceof Set ? seenSceneIds : new Set(seenSceneIds || []);
  const unseenTrials = [];
  const seenTrials = [];
  for (const trial of trials || []) {
    (seen.has(trial.scene_id) ? seenTrials : unseenTrials).push(trial);
  }
  return [...unseenTrials, ...seenTrials];
}

function buildTrials(scenes, options = {}) {
  const dataset = options.dataset || "all";
  const pairMode = options.pairMode || "compact";
  const seed = canonicalSeed(options.seed);
  const preserveSceneOrder = seed === SCENE_ORDER_SEED;
  const rng = createRng(seed);
  const trials = [];
  const sceneSequence = preserveSceneOrder
    ? (scenes || []).slice().sort((a, b) =>
        numberValue(a.manifest_order) - numberValue(b.manifest_order))
    : (scenes || []);

  for (const scene of sceneSequence) {
    if (dataset !== "all" && scene.dataset !== dataset) continue;
    const pairs = pairMode === COLOR_FIX_COVERAGE_MODE
      ? colorFixCoveragePairs(scene, seed)
      : candidatePairs(scene.candidates, pairMode);
    for (const [first, second] of pairs) {
      const swap = rng() < 0.5;
      const a = swap ? second : first;
      const b = swap ? first : second;
      trials.push({
        pair_id: canonicalPairId(scene.dataset, scene.scene_id, a.method, b.method),
        trial_id: `${scene.scene_id}__${a.method}__vs__${b.method}`,
        dataset: scene.dataset,
        scene_id: scene.scene_id,
        sample_id: scene.sample_id,
        input: scene.input,
        a,
        b,
      });
    }
  }

  const orderedTrials = preserveSceneOrder ? trials : shuffle(trials, rng);
  const prioritizedTrials = pairMode === COLOR_FIX_COVERAGE_MODE
    ? prioritizeUnseenSceneTrials(orderedTrials, options.seenSceneIds)
    : orderedTrials;
  const maxTrials = numberValue(options.maxTrials);
  return maxTrials > 0 ? prioritizedTrials.slice(0, maxTrials) : prioritizedTrials;
}

function createDisjointSet() {
  const parent = new Map(LABELS.map((label) => [label, label]));
  const find = (label) => {
    const p = parent.get(label);
    if (p === label) return label;
    const root = find(p);
    parent.set(label, root);
    return root;
  };
  const union = (a, b) => {
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent.set(rootB, rootA);
  };
  return { find, union };
}

function relationToLabels(key, value) {
  const pairs = {
    input_vs_a: ["Input", "A"],
    input_vs_b: ["Input", "B"],
    a_vs_b: ["A", "B"],
  };
  const pair = pairs[key];
  if (!pair || !value) return null;
  if (value === "similar") return { type: "similar", left: pair[0], right: pair[1] };
  if (value === "left") return { type: "strict", winner: pair[0], loser: pair[1] };
  if (value === "right") return { type: "strict", winner: pair[1], loser: pair[0] };
  return null;
}

function summarizeRanking(choices) {
  const dsu = createDisjointSet();
  const relations = [];
  for (const key of ["input_vs_a", "input_vs_b", "a_vs_b"]) {
    const relation = relationToLabels(key, choices?.[key]);
    if (!relation) continue;
    relations.push(relation);
    if (relation.type === "similar") dsu.union(relation.left, relation.right);
  }

  const groupsByRoot = new Map();
  for (const label of LABELS) {
    const root = dsu.find(label);
    if (!groupsByRoot.has(root)) groupsByRoot.set(root, []);
    groupsByRoot.get(root).push(label);
  }

  const groupId = new Map();
  const groups = Array.from(groupsByRoot.values());
  groups.forEach((group, index) => {
    for (const label of group) groupId.set(label, index);
  });

  const edges = new Map(groups.map((_, index) => [index, new Set()]));
  const indegree = new Map(groups.map((_, index) => [index, 0]));
  let hasConflict = false;

  for (const relation of relations) {
    if (relation.type !== "strict") continue;
    const from = groupId.get(relation.winner);
    const to = groupId.get(relation.loser);
    if (from === to) {
      hasConflict = true;
      continue;
    }
    if (!edges.get(from).has(to)) {
      edges.get(from).add(to);
      indegree.set(to, indegree.get(to) + 1);
    }
  }

  const ready = Array.from(indegree.entries())
    .filter(([, degree]) => degree === 0)
    .map(([id]) => id)
    .sort((a, b) => a - b);
  const ordered = [];
  while (ready.length > 0) {
    const id = ready.shift();
    ordered.push(id);
    for (const next of edges.get(id)) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) {
        ready.push(next);
        ready.sort((a, b) => a - b);
      }
    }
  }

  if (ordered.length !== groups.length) hasConflict = true;
  const rankedGroups = (hasConflict ? groups.map((_, index) => index) : ordered).map((id) => groups[id]);
  return {
    hasConflict,
    groups: rankedGroups,
    text: hasConflict ? "conflict" : rankedGroups.map((group) => group.join("=")).join(">"),
  };
}

function csvEscape(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const CSV_HEADER = [
  "expert_id",
  "group_id",
  "pair_id",
  "trial_id",
  "dataset",
  "scene_id",
  "input_method",
  "candidate_a_method",
  "candidate_b_method",
  "input_path",
  "candidate_a_path",
  "candidate_b_path",
  "input_vs_a",
  "input_vs_b",
  "a_vs_b",
  "ranking",
  "elapsed_ms",
  "timestamp",
];

function resultToCsvRow(result) {
  return CSV_HEADER.map((field) => csvEscape(result?.[field])).join(",");
}

function invertComparisonChoice(choice) {
  if (choice === "left") return "right";
  if (choice === "right") return "left";
  return choice;
}

function choicesForTrial(result, trial) {
  if (!result || !trial) return {};
  const choices = {
    input_vs_a: result.input_vs_a,
    input_vs_b: result.input_vs_b,
    a_vs_b: result.a_vs_b,
  };
  const savedMethods = [result.candidate_a_method, result.candidate_b_method];
  const displayedMethods = [trial.a?.method, trial.b?.method];
  if (![...savedMethods, ...displayedMethods].every(Boolean)) return {};
  if (savedMethods[0] === displayedMethods[0] && savedMethods[1] === displayedMethods[1]) {
    return choices;
  }
  if (savedMethods[0] === displayedMethods[1] && savedMethods[1] === displayedMethods[0]) {
    return {
      input_vs_a: choices.input_vs_b,
      input_vs_b: choices.input_vs_a,
      a_vs_b: invertComparisonChoice(choices.a_vs_b),
    };
  }
  return {};
}

const MosCore = {
  COLOR_FIX_COVERAGE_MODE,
  COLOR_FIX_METHOD,
  CSV_HEADER,
  SCENE_ORDER_SEED,
  buildScenes,
  buildTrials,
  canonicalPairId,
  canonicalSeed,
  choicesForTrial,
  csvEscape,
  imageOrientation,
  invertComparisonChoice,
  prioritizeUnseenSceneTrials,
  resultToCsvRow,
  summarizeRanking,
};

if (root) {
  root.MosCore = MosCore;
}

if (typeof module !== "undefined") {
  module.exports = MosCore;
}
})(typeof window !== "undefined" ? window : globalThis);
