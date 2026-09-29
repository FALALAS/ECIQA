import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  COLOR_FIX_COVERAGE_MODE,
  COLOR_FIX_METHOD,
  CSV_HEADER,
  buildScenes,
  buildTrials,
  canonicalPairId,
  canonicalSeed,
  choicesForTrial,
  imageOrientation,
  invertComparisonChoice,
  summarizeRanking,
  resultToCsvRow,
} = require("../label_core.js");

const records = [
  {
    dataset: "MSEC",
    scene_id: "scene_1",
    sample_id: "scene_1",
    item_type: "input",
    method: "input",
    display_order: 0,
    path: "../scenes/scene_1/00_input.jpg",
    width: 100,
    height: 80,
  },
  {
    dataset: "MSEC",
    scene_id: "scene_1",
    sample_id: "scene_1",
    item_type: "candidate",
    method: "cotf_original",
    display_order: 10,
    path: "../scenes/scene_1/10_cotf_original.png",
    width: 100,
    height: 80,
  },
  {
    dataset: "MSEC",
    scene_id: "scene_1",
    sample_id: "scene_1",
    item_type: "candidate",
    method: "fecnet_original",
    display_order: 20,
    path: "../scenes/scene_1/20_fecnet_original.jpg",
    width: 101,
    height: 81,
  },
  {
    dataset: "MSEC",
    scene_id: "scene_2",
    sample_id: "scene_2",
    item_type: "input",
    method: "input",
    display_order: 0,
    path: "../scenes/scene_2/00_input.jpg",
    width: 100,
    height: 80,
  },
];

const scenes = buildScenes(records);
assert.equal(scenes.length, 1, "scenes without two candidates are skipped");
assert.equal(scenes[0].candidates.length, 2);

const trials = buildTrials(scenes, {
  dataset: "MSEC",
  maxTrials: 3,
  seed: "tester-a",
  pairMode: "all",
});
assert.equal(trials.length, 1);
assert.equal(trials[0].input.method, "input");
assert.notEqual(trials[0].a.method, trials[0].b.method);
assert.equal(
  canonicalPairId("MSEC", "scene_1", "fecnet_original", "cotf_original"),
  "MSEC::scene_1::cotf_original::fecnet_original",
  "pair identity must not depend on Candidate A/B order",
);

assert.equal(invertComparisonChoice("left"), "right");
assert.equal(invertComparisonChoice("right"), "left");
assert.equal(invertComparisonChoice("similar"), "similar");
const savedChoices = Object.freeze({
  candidate_a_method: "cotf_original",
  candidate_b_method: "fecnet_original",
  input_vs_a: "left",
  input_vs_b: "right",
  a_vs_b: "right",
});
const originalTrial = {
  a: { method: "cotf_original" },
  b: { method: "fecnet_original" },
};
const swappedTrial = { a: originalTrial.b, b: originalTrial.a };
assert.deepEqual(
  choicesForTrial(savedChoices, originalTrial),
  { input_vs_a: "left", input_vs_b: "right", a_vs_b: "right" },
  "saved choices must be restored unchanged when A/B orientation matches",
);
const swappedChoices = choicesForTrial(savedChoices, swappedTrial);
assert.deepEqual(
  swappedChoices,
  { input_vs_a: "right", input_vs_b: "left", a_vs_b: "left" },
  "restoring after a seed swaps A/B must preserve image preferences",
);
assert.deepEqual(
  summarizeRanking(swappedChoices).groups,
  [["A"], ["Input"], ["B"]],
  "the preferred FECNet image must remain first after changing its display label",
);
assert.deepEqual(
  choicesForTrial({ ...savedChoices, a_vs_b: "similar" }, swappedTrial),
  { input_vs_a: "right", input_vs_b: "left", a_vs_b: "similar" },
  "an A/B tie must stay a tie when candidates are swapped",
);
assert.deepEqual(
  choicesForTrial({
    ...swappedChoices,
    candidate_a_method: swappedTrial.a.method,
    candidate_b_method: swappedTrial.b.method,
  }, originalTrial),
  choicesForTrial(savedChoices, originalTrial),
  "restoring twice across opposite orientations must recover the original choices",
);
assert.deepEqual(
  choicesForTrial(savedChoices, { a: { method: "unknown" }, b: originalTrial.a }),
  {},
  "unrelated saved candidates must not prefill a trial",
);
assert.deepEqual(choicesForTrial(null, originalTrial), {});
assert.deepEqual(choicesForTrial(savedChoices, null), {});
assert.deepEqual(choicesForTrial({ input_vs_a: "left" }, { a: {}, b: {} }), {});

assert.equal(
  trials[0].pair_id,
  canonicalPairId(
    trials[0].dataset,
    trials[0].scene_id,
    trials[0].a.method,
    trials[0].b.method,
  ),
  "every trial must expose its canonical pair identity",
);

assert.deepEqual(
  summarizeRanking({
    input_vs_a: "left",
    input_vs_b: "left",
    a_vs_b: "similar",
  }).groups,
  [["Input"], ["A", "B"]],
);

assert.equal(
  summarizeRanking({
    input_vs_a: "left",
    input_vs_b: "right",
    a_vs_b: "left",
  }).hasConflict,
  true,
  "cyclic strict preferences should be rejected",
);

const row = resultToCsvRow({
  expert_id: "e,1",
  group_id: "A",
  pair_id: "MSEC::scene_1::cotf_original::fecnet_original",
  trial_id: "t1",
  dataset: "MSEC",
  scene_id: "scene_1",
  input_method: "input",
  candidate_a_method: "cotf_original",
  candidate_b_method: "fecnet_original",
  input_path: "../scenes/scene_1/00_input.jpg",
  candidate_a_path: "../scenes/scene_1/10.png",
  candidate_b_path: "../scenes/scene_1/20.jpg",
  input_vs_a: "left",
  input_vs_b: "similar",
  a_vs_b: "right",
  ranking: "Input>A=B",
  elapsed_ms: 1234,
  timestamp: "2026-07-22T12:00:00.000Z",
});
assert.deepEqual(CSV_HEADER.slice(0, 4), ["expert_id", "group_id", "pair_id", "trial_id"]);
assert.match(row, /^"e,1",A,MSEC::scene_1::cotf_original::fecnet_original,t1,MSEC/);
assert.match(row, /Input>A=B/);

function record(dataset, sceneId, method, itemType = "candidate", displayOrder = 10) {
  return {
    dataset,
    scene_id: sceneId,
    sample_id: sceneId,
    item_type: itemType,
    method,
    display_order: displayOrder,
    path: `../scenes/${sceneId}/${method}.jpg`,
    width: 100,
    height: 80,
  };
}

function normalizedPairs(trials) {
  return trials
    .map((trial) => [trial.a.method, trial.b.method].sort().join("::"))
    .sort();
}

assert.equal(canonicalSeed(" SCENE-ORDER "), "scene-order");
assert.equal(canonicalSeed(" subject-07 "), "subject-07");
assert.equal(imageOrientation({ width: 1200, height: 800 }), "landscape");
assert.equal(imageOrientation({ width: 800, height: 1200 }), "portrait");
assert.equal(imageOrientation({ width: 900, height: 900 }), "landscape");
assert.equal(imageOrientation({ width: 0, height: 0 }), "landscape");

const sourceOrderedRecords = [
  record("MSEC", "scene_z", "input", "input", 0),
  record("MSEC", "scene_z", "cotf_original", "candidate", 10),
  record("MSEC", "scene_z", "fecnet_original", "candidate", 20),
  record("MSEC", "scene_z", "fecnet_mefssim_1x", "candidate", 30),
  record("MSEC", "scene_a", "input", "input", 0),
  record("MSEC", "scene_a", "cotf_original", "candidate", 10),
  record("MSEC", "scene_a", "fecnet_original", "candidate", 20),
  record("MSEC", "scene_a", "fecnet_mefssim_1x", "candidate", 30),
];
const sourceOrderedScenes = buildScenes(sourceOrderedRecords);
assert.deepEqual(
  sourceOrderedScenes.map(({ scene_id }) => scene_id),
  ["scene_a", "scene_z"],
  "ordinary scene preparation must retain its existing scene-id sort",
);
assert.deepEqual(
  Object.fromEntries(
    sourceOrderedScenes.map(({ scene_id, manifest_order }) => [scene_id, manifest_order]),
  ),
  { scene_a: 1, scene_z: 0 },
  "each scene must retain its first-occurrence order from the manifest",
);

const ordinaryBaseline = buildTrials(sourceOrderedScenes, {
  dataset: "MSEC",
  pairMode: "all",
  seed: "ordinary-baseline",
});
assert.deepEqual(
  ordinaryBaseline.map(({ scene_id, a, b }) => `${scene_id}:${a.method}>${b.method}`),
  [
    "scene_a:fecnet_original>cotf_original",
    "scene_z:fecnet_original>fecnet_mefssim_1x",
    "scene_a:fecnet_original>fecnet_mefssim_1x",
    "scene_a:cotf_original>fecnet_mefssim_1x",
    "scene_z:cotf_original>fecnet_mefssim_1x",
    "scene_z:fecnet_original>cotf_original",
  ],
  "ordinary seeds must retain the pre-feature shuffle and side assignment",
);

const sceneOrdered = buildTrials(sourceOrderedScenes, {
  dataset: "MSEC",
  pairMode: "all",
  seed: "SCENE-ORDER",
});
assert.deepEqual(
  sceneOrdered.map(({ scene_id }) => scene_id),
  ["scene_z", "scene_z", "scene_z", "scene_a", "scene_a", "scene_a"],
  "scene-order must group trials by manifest scene order",
);
assert.deepEqual(
  sceneOrdered.map(({ a, b }) => [a.method, b.method].sort().join("::")),
  [
    "cotf_original::fecnet_original",
    "cotf_original::fecnet_mefssim_1x",
    "fecnet_mefssim_1x::fecnet_original",
    "cotf_original::fecnet_original",
    "cotf_original::fecnet_mefssim_1x",
    "fecnet_mefssim_1x::fecnet_original",
  ],
  "scene-order must retain candidatePairs order within every scene",
);
assert.deepEqual(
  sceneOrdered,
  buildTrials(sourceOrderedScenes, {
    dataset: "MSEC",
    pairMode: "all",
    seed: "scene-order",
  }),
  "scene-order A/B swaps must be reproducible and case-insensitive",
);
assert.deepEqual(
  buildTrials(sourceOrderedScenes, {
    dataset: "MSEC",
    pairMode: "all",
    seed: "scene-order",
    maxTrials: 4,
  }),
  sceneOrdered.slice(0, 4),
  "scene-order trial limits must take the first N ordered trials",
);

const siceSceneWithoutReferences = buildScenes([
  record("SICEV2", "sice_references", "input", "input", 0),
  record("SICEV2", "sice_references", "middle_gt", "reference", 1),
  record("SICEV2", "sice_references", "fused_gt", "reference", 2),
  record("SICEV2", "sice_references", "cotf_original", "candidate", 10),
  record("SICEV2", "sice_references", "fecnet_original", "candidate", 20),
]);
assert.deepEqual(
  siceSceneWithoutReferences[0].candidates.map(({ method }) => method),
  ["cotf_original", "fecnet_original"],
  "reference images must never enter ranked candidates",
);

const msecMethods = [
  "cotf_original",
  "cotf_mefssim_0.5x_color_fix",
  "fecnet_original",
  "fecnet_mefssim_0.5x",
];
const msecCompactScene = buildScenes([
  record("MSEC", "msec_compact", "input", "input", 0),
  ...msecMethods.map((method, index) =>
    record("MSEC", "msec_compact", method, "candidate", 10 + index),
  ),
]);
assert.deepEqual(
  normalizedPairs(
    buildTrials(msecCompactScene, {
      dataset: "MSEC",
      pairMode: "compact",
      seed: "pairs",
    }),
  ),
  [
    "cotf_mefssim_0.5x_color_fix::cotf_original",
    "cotf_mefssim_0.5x_color_fix::fecnet_mefssim_0.5x",
    "cotf_original::fecnet_original",
    "fecnet_mefssim_0.5x::fecnet_original",
  ].sort(),
  "MSEC Compact mode must contain the four approved pairs",
);

const coverageSceneIds = ["coverage_z", "coverage_a", "coverage_m"];
const coverageScenes = buildScenes(coverageSceneIds.flatMap((sceneId) => [
  record("MSEC", sceneId, "input", "input", 0),
  ...msecMethods.map((method, index) =>
    record("MSEC", sceneId, method, "candidate", 10 + index)),
]));
const coverageBaseline = buildTrials(coverageScenes, {
  dataset: "MSEC",
  pairMode: COLOR_FIX_COVERAGE_MODE,
  seed: "scene-order",
});
assert.equal(coverageBaseline.length, coverageSceneIds.length);
assert.deepEqual(
  coverageBaseline.map(({ scene_id }) => scene_id),
  coverageSceneIds,
  "color-fix coverage must create one trial per scene in manifest order",
);
for (const trial of coverageBaseline) {
  const methods = [trial.a.method, trial.b.method];
  assert.equal(methods.includes(COLOR_FIX_METHOD), true);
  assert.equal(new Set(methods).size, 2);
}
assert.deepEqual(
  buildTrials(coverageScenes, {
    dataset: "MSEC",
    pairMode: COLOR_FIX_COVERAGE_MODE,
    seed: "scene-order",
    seenSceneIds: new Set(["coverage_z", "coverage_m"]),
  }).map(({ scene_id }) => scene_id),
  ["coverage_a", "coverage_z", "coverage_m"],
  "globally unseen scenes must be stably moved ahead of seen scenes",
);
const possibleOpponents = new Set();
for (let index = 0; index < 100; index += 1) {
  const trial = buildTrials([coverageScenes[0]], {
    dataset: "MSEC",
    pairMode: COLOR_FIX_COVERAGE_MODE,
    seed: `opponent-${index}`,
  })[0];
  possibleOpponents.add(
    trial.a.method === COLOR_FIX_METHOD ? trial.b.method : trial.a.method,
  );
}
assert.deepEqual(
  Array.from(possibleOpponents).sort(),
  ["cotf_original", "fecnet_mefssim_0.5x", "fecnet_original"].sort(),
  "the color-fix opponent pool must include every other MSEC model output",
);

const siceCompactScene = buildScenes([
  record("SICEV2", "sice_compact", "input", "input", 0),
  record("SICEV2", "sice_compact", "middle_gt", "reference", 1),
  record("SICEV2", "sice_compact", "fused_gt", "reference", 2),
  record("SICEV2", "sice_compact", "cotf_original", "candidate", 10),
  record("SICEV2", "sice_compact", "cotf_fusegt", "candidate", 12),
  record("SICEV2", "sice_compact", "fecnet_original", "candidate", 20),
  record("SICEV2", "sice_compact", "fecnet_fusegt", "candidate", 23),
]);
assert.deepEqual(
  normalizedPairs(
    buildTrials(siceCompactScene, {
      dataset: "SICEV2",
      pairMode: "compact",
      seed: "pairs",
    }),
  ),
  [
    "cotf_fusegt::cotf_original",
    "cotf_fusegt::fecnet_fusegt",
    "cotf_original::fecnet_original",
    "fecnet_fusegt::fecnet_original",
  ].sort(),
  "SICEV2 Compact mode must contain the four approved model-output pairs",
);
assert.equal(
  buildTrials(siceCompactScene, {
    dataset: "SICEV2",
    pairMode: "all",
    seed: "pairs",
  }).length,
  6,
  "All-pairs mode must use only the four SICEV2 model outputs",
);
assert.equal(
  buildTrials(siceCompactScene, {
    dataset: "SICEV2",
    pairMode: COLOR_FIX_COVERAGE_MODE,
    seed: "coverage",
  }).length,
  0,
  "color-fix coverage must skip datasets without a color-fix candidate",
);

console.log("label_core tests passed");
