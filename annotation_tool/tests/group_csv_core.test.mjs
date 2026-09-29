import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { CSV_HEADER, resultToCsvRow } = require("../label_core.js");

class MemoryFileHandle {
  constructor(name, files) {
    this.name = name;
    this.files = files;
  }

  async getFile() {
    const bytes = this.files.get(this.name) || Buffer.alloc(0);
    return {
      size: bytes.length,
      text: async () => bytes.toString("utf8"),
    };
  }

  async createWritable(options = {}) {
    let bytes = options.keepExistingData
      ? Buffer.from(this.files.get(this.name) || Buffer.alloc(0))
      : Buffer.alloc(0);
    let position = 0;
    return {
      seek: async (nextPosition) => {
        position = Number(nextPosition);
      },
      write: async (data) => {
        const incoming = Buffer.from(data);
        const required = position + incoming.length;
        if (required > bytes.length) {
          bytes = Buffer.concat([bytes, Buffer.alloc(required - bytes.length)]);
        }
        incoming.copy(bytes, position);
        position += incoming.length;
      },
      close: async () => {
        this.files.set(this.name, bytes);
      },
    };
  }
}

class MemoryDirectoryHandle {
  constructor(initialFiles = {}) {
    this.name = "mos-handoff";
    this.files = new Map(
      Object.entries(initialFiles).map(([name, text]) => [name, Buffer.from(text, "utf8")]),
    );
  }

  async getFileHandle(name, options = {}) {
    if (!this.files.has(name)) {
      if (!options.create) throw new DOMException(`Missing file: ${name}`, "NotFoundError");
      this.files.set(name, Buffer.alloc(0));
    }
    return new MemoryFileHandle(name, this.files);
  }

  names() {
    return Array.from(this.files.keys()).sort();
  }

  text(name) {
    return (this.files.get(name) || Buffer.alloc(0)).toString("utf8");
  }
}

function result(overrides = {}) {
  return {
    expert_id: "e01",
    group_id: "A",
    pair_id: "MSEC::scene_1::cotf_original::fecnet_original",
    trial_id: "scene_1__cotf_original__vs__fecnet_original",
    dataset: "MSEC",
    scene_id: "scene_1",
    input_method: "input",
    candidate_a_method: "cotf_original",
    candidate_b_method: "fecnet_original",
    input_path: "../scenes/scene_1/00_input.jpg",
    candidate_a_path: "../scenes/scene_1/10_cotf.png",
    candidate_b_path: "../scenes/scene_1/20_fecnet.jpg",
    input_vs_a: "left",
    input_vs_b: "left",
    a_vs_b: "similar",
    ranking: "Input>A=B",
    elapsed_ms: 1000,
    timestamp: "2026-07-29T00:00:00.000Z",
    ...overrides,
  };
}

function trial(overrides = {}) {
  return {
    pair_id: "MSEC::scene_1::cotf_original::fecnet_original",
    trial_id: "scene_1__cotf_original__vs__fecnet_original",
    dataset: "MSEC",
    scene_id: "scene_1",
    input: {
      method: "input",
      path: "../scenes/scene_1/00_input.jpg",
      width: 100,
      height: 80,
    },
    a: {
      method: "cotf_original",
      path: "../scenes/scene_1/10_cotf.png",
      width: 100,
      height: 80,
    },
    b: {
      method: "fecnet_original",
      path: "../scenes/scene_1/20_fecnet.jpg",
      width: 100,
      height: 80,
    },
    ...overrides,
  };
}

const {
  COMPARISON_MARK_FILENAME,
  COMPARISON_MARK_HEADER,
  GROUP_IDS,
  REVIEW_HEADER,
  annotationReviewId,
  comparisonMarksToTrials,
  createGroupCsvRepository,
  isAnnotationReviewed,
  isComparisonMarked,
  latestResultsByPair,
  parseCsv,
  reviewGroupFilename,
  selectUnfinishedTrials,
} = require("../group_csv_core.js");

assert.deepEqual(GROUP_IDS, ["A", "B", "C", "D", "E"]);
assert.deepEqual(
  parseCsv('expert_id,ranking\r\n"e,1","Input>A=B"\r\n'),
  [{ expert_id: "e,1", ranking: "Input>A=B" }],
  "CSV parser must support quoted commas and CRLF",
);
assert.deepEqual(
  parseCsv('expert_id,ranking\n"e""2","Input\nA"\n'),
  [{ expert_id: 'e"2', ranking: "Input\nA" }],
  "CSV parser must support escaped quotes and quoted newlines",
);

const oldResult = result({ pair_id: undefined });
const normalizedOld = latestResultsByPair([oldResult]);
assert.equal(normalizedOld.size, 1);
assert.equal(
  normalizedOld.get("MSEC::scene_1::cotf_original::fecnet_original").pair_id,
  "MSEC::scene_1::cotf_original::fecnet_original",
  "old rows must derive canonical pair identity",
);

const duplicateLatest = latestResultsByPair([
  result({ ranking: "Input>A>B", timestamp: "2026-07-29T00:00:00.000Z" }),
  result({ ranking: "B>Input>A", timestamp: "2026-07-29T00:01:00.000Z" }),
]);
assert.equal(duplicateLatest.size, 1);
assert.equal(duplicateLatest.values().next().value.ranking, "B>Input>A");

assert.deepEqual(
  selectUnfinishedTrials(
    [{ pair_id: "p1" }, { pair_id: "p2" }, { pair_id: "p3" }],
    new Map([["p1", {}]]),
    1,
  ),
  [{ pair_id: "p2" }],
  "allocation must take the first N unfinished pairs",
);

const directory = new MemoryDirectoryHandle();
const repository = createGroupCsvRepository(directory);
await repository.initialize();
assert.deepEqual(directory.names(), [
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
]);
for (const groupId of GROUP_IDS) {
  assert.equal(
    directory.text(`group_${groupId}.csv`),
    `${CSV_HEADER.join(",")}\r\n`,
    `Group ${groupId} must be initialized with the current header`,
  );
  assert.equal(
    directory.text(reviewGroupFilename(groupId)),
    `${REVIEW_HEADER.join(",")}\r\n`,
    `Review Group ${groupId} must be initialized with the review header`,
  );
}
assert.equal(
  directory.text(COMPARISON_MARK_FILENAME),
  `${COMPARISON_MARK_HEADER.join(",")}\r\n`,
  "the comparison mark file must be initialized with its own header",
);

const sourceAnnotation = result();
const annotationId = annotationReviewId(sourceAnnotation);
assert.ok(annotationId.includes(sourceAnnotation.pair_id));
assert.equal(isAnnotationReviewed(new Map(), sourceAnnotation), false);
const reviewedAnnotation = result({
  a_vs_b: "left",
  ranking: "Input>A>B",
});
const firstReview = await repository.setReviewed({
  reviewer_id: "reviewer-01",
  review_group_id: "A",
  source_result: sourceAnnotation,
  review_result: reviewedAnnotation,
  reviewed: true,
  reviewed_at: "2026-08-06T00:00:00.000Z",
});
assert.equal(firstReview.affectedRows, 1);
assert.equal(firstReview.reviewed, true);
assert.equal(isAnnotationReviewed(firstReview.reviews, sourceAnnotation), true);
let reviews = await repository.readReviewGroup("A");
assert.equal(reviews.size, 1);
assert.equal(reviews.get(annotationId).source_group_id, "A");
assert.equal(reviews.get(annotationId).source_expert_id, "e01");
assert.equal(reviews.get(annotationId).reviewer_id, "reviewer-01");
assert.equal(reviews.get(annotationId).a_vs_b, "left");
assert.equal(reviews.get(annotationId).ranking, "Input>A>B");
assert.equal(reviews.get(annotationId).source_timestamp, sourceAnnotation.timestamp);
assert.equal(parseCsv(directory.text("group_A_reviewed.csv")).length, 1);

const updatedReview = await repository.setReviewed({
  reviewer_id: "reviewer-02",
  review_group_id: "A",
  source_result: sourceAnnotation,
  review_result: result({ a_vs_b: "right", ranking: "Input>B>A" }),
  reviewed: true,
  reviewed_at: "2026-08-06T00:00:30.000Z",
});
assert.equal(updatedReview.reviews.size, 1);
assert.equal(updatedReview.reviews.get(annotationId).a_vs_b, "right");
assert.equal(updatedReview.reviews.get(annotationId).ranking, "Input>B>A");
assert.equal(updatedReview.reviews.get(annotationId).reviewer_id, "reviewer-02");
assert.equal(parseCsv(directory.text("group_A_reviewed.csv")).length, 1);

await assert.rejects(
  repository.setReviewed({
    reviewer_id: "reviewer-03",
    review_group_id: "A",
    source_result: sourceAnnotation,
    review_result: result({
      input_vs_a: "left",
      input_vs_b: "right",
      a_vs_b: "left",
    }),
    reviewed: true,
  }),
  /complete, consistent ranking/,
);
reviews = await repository.readReviewGroup("A");
assert.equal(reviews.get(annotationId).a_vs_b, "right", "a rejected edit must preserve the saved review");

const correctedAnnotation = result({
  input_vs_a: "right",
  a_vs_b: "left",
  ranking: "A>Input>B",
  timestamp: "2026-08-06T00:01:00.000Z",
});
assert.notEqual(
  annotationReviewId(correctedAnnotation),
  annotationId,
  "a corrected source result must require a new review",
);
assert.equal(isAnnotationReviewed(reviews, correctedAnnotation), false);

const removedReview = await repository.setReviewed({
  reviewer_id: "reviewer-01",
  review_group_id: "A",
  source_result: sourceAnnotation,
  reviewed: false,
});
assert.equal(removedReview.reviewed, false);
assert.equal(removedReview.affectedRows, 1);
reviews = await repository.readReviewGroup("A");
assert.equal(reviews.size, 0);
assert.equal(parseCsv(directory.text("group_A_reviewed.csv")).length, 0);

const firstTrial = trial();
const firstMark = await repository.setComparisonMarked({
  expert_id: "e01",
  group_id: "A",
  trial: firstTrial,
  marked: true,
  marked_at: "2026-07-31T00:00:00.000Z",
});
assert.equal(firstMark.affectedRows, 1);
assert.equal(isComparisonMarked(firstMark.marks, firstTrial.pair_id), true);
let comparisonMarks = await repository.readComparisonMarks();
assert.equal(comparisonMarks.size, 1);
const firstMarkRows = parseCsv(directory.text(COMPARISON_MARK_FILENAME));
assert.equal(firstMarkRows.length, 1);
assert.equal(firstMarkRows[0].mark_order, "1");
assert.equal(firstMarkRows[0].candidate_a_path, firstTrial.a.path);
assert.equal(firstMarkRows[0].candidate_b_path, firstTrial.b.path);

const restored = comparisonMarksToTrials(comparisonMarks, [
  firstTrial.input,
  firstTrial.a,
  firstTrial.b,
]);
assert.equal(restored.length, 1);
assert.equal(restored[0].pair_id, firstTrial.pair_id);
assert.equal(restored[0].a.path, firstTrial.a.path);
assert.equal(restored[0].b.path, firstTrial.b.path);
assert.equal(restored[0].a.width, 100, "restoration should reuse manifest metadata when available");

await assert.rejects(
  repository.setComparisonMarked({
    expert_id: "e01",
    group_id: "A",
    trial: { pair_id: "incomplete" },
    marked: true,
  }),
  /missing reconstruction fields/,
);
comparisonMarks = await repository.readComparisonMarks();
assert.equal(comparisonMarks.size, 1, "a rejected mark must preserve the existing file and cache");

const secondTrial = trial({
  pair_id: "MSEC::scene_2::cotf_original::fecnet_original",
  trial_id: "scene_2__cotf_original__vs__fecnet_original",
  scene_id: "scene_2",
  input: { method: "input", path: "../scenes/scene_2/00_input.jpg" },
  a: { method: "cotf_original", path: "../scenes/scene_2/10_cotf.png" },
  b: { method: "fecnet_original", path: "../scenes/scene_2/20_fecnet.jpg" },
});
await repository.setComparisonMarked({
  expert_id: "e02",
  group_id: "B",
  trial: secondTrial,
  marked: true,
  marked_at: "2026-07-31T00:02:00.000Z",
});
comparisonMarks = await repository.readComparisonMarks();
assert.deepEqual(
  comparisonMarksToTrials(comparisonMarks, []).map(({ pair_id }) => pair_id),
  [firstTrial.pair_id, secondTrial.pair_id],
  "marked-only restoration must follow mark_order",
);

const unmarked = await repository.setComparisonMarked({
  expert_id: "e01",
  group_id: "A",
  trial: firstTrial,
  marked: false,
});
assert.equal(unmarked.affectedRows, 1);
assert.equal(isComparisonMarked(unmarked.marks, firstTrial.pair_id), false);
assert.equal(isComparisonMarked(unmarked.marks, secondTrial.pair_id), true);
comparisonMarks = await repository.readComparisonMarks();
assert.equal(comparisonMarks.size, 1);

const firstSave = await repository.saveResult("A", result());
assert.equal(firstSave.mode, "append");
let stored = await repository.readGroup("A");
assert.equal(stored.size, 1);
assert.equal(stored.values().next().value.expert_id, "e01");

const correction = await repository.saveResult("A", result({
  ranking: "B>Input=A",
  timestamp: "2026-07-29T00:02:00.000Z",
}));
assert.equal(correction.mode, "rewrite");
stored = await repository.readGroup("A");
assert.equal(stored.size, 1);
assert.equal(stored.values().next().value.ranking, "B>Input=A");
assert.equal(
  directory.text("group_A.csv").trim().split(/\r?\n/).length,
  2,
  "a corrected pair must still occupy one data row",
);

const localNewer = result({
  expert_id: "e02",
  ranking: "A>B>Input",
  timestamp: "2026-07-29T00:03:00.000Z",
});
const merged = await repository.mergeResults("A", [localNewer]);
assert.equal(merged.changed, true);
assert.equal(merged.results.get(localNewer.pair_id).expert_id, "e02");

const localOlder = result({
  expert_id: "e00",
  ranking: "Input>B>A",
  timestamp: "2026-07-28T00:00:00.000Z",
});
const unchanged = await repository.mergeResults("A", [localOlder]);
assert.equal(unchanged.changed, false);
assert.equal(unchanged.results.get(localOlder.pair_id).expert_id, "e02");

const legacyHeader = CSV_HEADER.filter((field) => field !== "pair_id");
const legacyRow = legacyHeader.map((field) => oldResult[field] ?? "").join(",");
const legacyDirectory = new MemoryDirectoryHandle({
  "group_B.csv": `${legacyHeader.join(",")}\r\n${legacyRow}\r\n`,
});
const legacyRepository = createGroupCsvRepository(legacyDirectory);
await legacyRepository.initialize();
const upgraded = await legacyRepository.readGroup("B");
assert.equal(upgraded.size, 1);
assert.equal(upgraded.values().next().value.group_id, "B");
assert.equal(legacyDirectory.text("group_B.csv").startsWith(`${CSV_HEADER.join(",")}\r\n`), true);

const duplicateDirectory = new MemoryDirectoryHandle({
  "group_D.csv": [
    CSV_HEADER.join(","),
    resultToCsvRow(result({ group_id: "D", ranking: "Input>A>B" })),
    resultToCsvRow(result({ group_id: "D", ranking: "B>Input>A" })),
    "",
  ].join("\r\n"),
});
const duplicateRepository = createGroupCsvRepository(duplicateDirectory);
await duplicateRepository.initialize();
const deduplicated = await duplicateRepository.readGroup("D");
assert.equal(deduplicated.size, 1);
assert.equal(deduplicated.values().next().value.ranking, "B>Input>A");
assert.equal(duplicateDirectory.text("group_D.csv").trim().split(/\r?\n/).length, 2);

const malformedDirectory = new MemoryDirectoryHandle({
  "group_C.csv": "wrong,columns\r\nkeep,this\r\n",
});
const malformedRepository = createGroupCsvRepository(malformedDirectory);
await assert.rejects(
  malformedRepository.initialize(),
  /Invalid CSV header in group_C\.csv/,
);
assert.equal(
  malformedDirectory.text("group_C.csv"),
  "wrong,columns\r\nkeep,this\r\n",
  "malformed files must not be overwritten",
);

console.log("group CSV core tests passed");
