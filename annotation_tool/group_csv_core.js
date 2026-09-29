(function attachMosGroupCsv(root) {
const core = root?.MosCore || (typeof require !== "undefined" ? require("./label_core.js") : null);
if (!core) throw new Error("MosCore must be loaded before MosGroupCsv");

const {
  CSV_HEADER,
  canonicalPairId,
  csvEscape,
  resultToCsvRow,
  summarizeRanking,
} = core;
const GROUP_IDS = ["A", "B", "C", "D", "E"];
const LEGACY_HEADER = CSV_HEADER.filter((field) => field !== "pair_id");
const COMPARISON_MARK_FILENAME = "marked_comparisons.csv";
const COMPARISON_MARK_HEADER = [
  "mark_order",
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
  "marked_at",
];
const REVIEW_HEADER = [
  "review_order",
  "review_group_id",
  "reviewer_id",
  "source_group_id",
  "source_expert_id",
  "annotation_id",
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
  "source_elapsed_ms",
  "source_timestamp",
  "reviewed_at",
];
const encoder = new TextEncoder();

function groupFilename(groupId) {
  if (!GROUP_IDS.includes(groupId)) throw new Error(`Invalid Group: ${groupId}`);
  return `group_${groupId}.csv`;
}

function reviewGroupFilename(groupId) {
  if (!GROUP_IDS.includes(groupId)) throw new Error(`Invalid Review Group: ${groupId}`);
  return `group_${groupId}_reviewed.csv`;
}

function parseCsvDocument(text) {
  const input = String(text || "").replace(/^\uFEFF/, "");
  if (!input) return { header: [], rows: [] };

  const table = [];
  let row = [];
  let field = "";
  let quoted = false;

  const finishRow = () => {
    row.push(field);
    if (row.some((value) => value !== "")) table.push(row);
    row = [];
    field = "";
  };

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (quoted) {
      if (character === '"') {
        if (input[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += character;
      }
      continue;
    }

    if (character === '"' && field === "") {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\r" || character === "\n") {
      if (character === "\r" && input[index + 1] === "\n") index += 1;
      finishRow();
    } else {
      field += character;
    }
  }

  if (quoted) throw new Error("Unclosed quoted CSV field");
  if (field !== "" || row.length > 0) finishRow();
  if (table.length === 0) return { header: [], rows: [] };

  const [header, ...dataRows] = table;
  const records = dataRows.map((values, rowIndex) => {
    if (values.length > header.length) {
      throw new Error(`CSV row ${rowIndex + 2} has more fields than the header`);
    }
    return Object.fromEntries(header.map((name, column) => [name, values[column] ?? ""]));
  });
  return { header, rows: records };
}

function parseCsv(text) {
  return parseCsvDocument(text).rows;
}

function sameHeader(actual, expected) {
  return actual.length === expected.length && actual.every((field, index) => field === expected[index]);
}

function normalizeResult(result, groupId = result?.group_id) {
  if (!result || typeof result !== "object") return null;
  const normalizedGroup = GROUP_IDS.includes(groupId) ? groupId : String(result.group_id || "");
  const pairId = result.pair_id || canonicalPairId(
    result.dataset,
    result.scene_id,
    result.candidate_a_method,
    result.candidate_b_method,
  );
  if (!pairId || !result.dataset || !result.scene_id ||
      !result.candidate_a_method || !result.candidate_b_method) return null;

  const { subject_id: subjectId, ...rest } = result;
  return {
    ...rest,
    expert_id: String(rest.expert_id || subjectId || "anonymous").trim() || "anonymous",
    group_id: normalizedGroup,
    pair_id: pairId,
  };
}

function latestResultsByPair(results, groupId) {
  const latest = new Map();
  for (const result of results || []) {
    const normalized = normalizeResult(result, groupId || result?.group_id);
    if (normalized) latest.set(normalized.pair_id, normalized);
  }
  return latest;
}

function selectUnfinishedTrials(trials, completedByPair, maxTrials) {
  const unfinished = (trials || []).filter((trial) => !completedByPair.has(trial.pair_id));
  const limit = Number(maxTrials);
  return Number.isFinite(limit) && limit > 0 ? unfinished.slice(0, limit) : unfinished;
}

function serializeResults(results) {
  const rows = Array.from(results || [], (result) => resultToCsvRow(result));
  return `${CSV_HEADER.join(",")}\r\n${rows.length ? `${rows.join("\r\n")}\r\n` : ""}`;
}

function lengthPrefixedIdentity(parts) {
  return parts.map((part) => {
    const value = String(part ?? "");
    return `${value.length}:${value}`;
  }).join("|");
}

function annotationReviewId(result) {
  const normalized = normalizeResult(result, result?.group_id);
  if (!normalized || !GROUP_IDS.includes(normalized.group_id)) return "";
  return lengthPrefixedIdentity([
    normalized.group_id,
    normalized.pair_id,
    normalized.trial_id,
    normalized.candidate_a_method,
    normalized.candidate_b_method,
    normalized.input_vs_a,
    normalized.input_vs_b,
    normalized.a_vs_b,
    normalized.ranking,
    normalized.timestamp,
  ]);
}

function normalizeReview(review, reviewGroupId = review?.review_group_id) {
  if (!review || typeof review !== "object") return null;
  const normalizedReviewGroup = String(reviewGroupId || "").trim().toUpperCase();
  const sourceGroupId = String(review.source_group_id || "").trim().toUpperCase();
  const reviewOrder = Number(review.review_order);
  const sourceResult = normalizeResult({
    expert_id: review.source_expert_id,
    group_id: sourceGroupId,
    pair_id: review.pair_id,
    trial_id: review.trial_id,
    dataset: review.dataset,
    scene_id: review.scene_id,
    input_method: review.input_method,
    candidate_a_method: review.candidate_a_method,
    candidate_b_method: review.candidate_b_method,
    input_path: review.input_path,
    candidate_a_path: review.candidate_a_path,
    candidate_b_path: review.candidate_b_path,
    input_vs_a: review.input_vs_a,
    input_vs_b: review.input_vs_b,
    a_vs_b: review.a_vs_b,
    ranking: review.ranking,
    elapsed_ms: review.source_elapsed_ms,
    timestamp: review.source_timestamp,
  }, sourceGroupId);
  const annotationId = String(review.annotation_id || annotationReviewId(sourceResult)).trim();
  const reviewerId = String(review.reviewer_id || "anonymous").trim() || "anonymous";

  if (!GROUP_IDS.includes(normalizedReviewGroup) || !GROUP_IDS.includes(sourceGroupId) ||
      !Number.isInteger(reviewOrder) || reviewOrder < 1 || !sourceResult || !annotationId) {
    return null;
  }

  return {
    review_order: reviewOrder,
    review_group_id: normalizedReviewGroup,
    reviewer_id: reviewerId,
    source_group_id: sourceGroupId,
    source_expert_id: sourceResult.expert_id,
    annotation_id: annotationId,
    pair_id: sourceResult.pair_id,
    trial_id: String(sourceResult.trial_id || ""),
    dataset: sourceResult.dataset,
    scene_id: sourceResult.scene_id,
    input_method: String(sourceResult.input_method || "input"),
    candidate_a_method: sourceResult.candidate_a_method,
    candidate_b_method: sourceResult.candidate_b_method,
    input_path: String(sourceResult.input_path || ""),
    candidate_a_path: String(sourceResult.candidate_a_path || ""),
    candidate_b_path: String(sourceResult.candidate_b_path || ""),
    input_vs_a: String(sourceResult.input_vs_a || ""),
    input_vs_b: String(sourceResult.input_vs_b || ""),
    a_vs_b: String(sourceResult.a_vs_b || ""),
    ranking: String(sourceResult.ranking || ""),
    source_elapsed_ms: String(sourceResult.elapsed_ms ?? ""),
    source_timestamp: String(sourceResult.timestamp || ""),
    reviewed_at: String(review.reviewed_at || "").trim(),
  };
}

function latestReviewsByAnnotation(reviews, reviewGroupId) {
  const latest = new Map();
  for (const review of reviews || []) {
    const normalized = normalizeReview(review, reviewGroupId || review?.review_group_id);
    if (!normalized) continue;
    if (latest.has(normalized.annotation_id)) latest.delete(normalized.annotation_id);
    latest.set(normalized.annotation_id, normalized);
  }
  return latest;
}

function orderedReviews(reviews) {
  const values = reviews instanceof Map ? reviews.values() : reviews || [];
  return Array.from(values).sort((left, right) =>
    Number(left.review_order) - Number(right.review_order));
}

function reviewToCsvRow(review) {
  return REVIEW_HEADER.map((field) => csvEscape(review?.[field])).join(",");
}

function serializeReviews(reviews) {
  const rows = orderedReviews(reviews).map((review) => reviewToCsvRow(review));
  return `${REVIEW_HEADER.join(",")}\r\n${rows.length ? `${rows.join("\r\n")}\r\n` : ""}`;
}

function isAnnotationReviewed(reviews, annotationOrId) {
  const annotationId = typeof annotationOrId === "string"
    ? annotationOrId
    : annotationReviewId(annotationOrId);
  if (!annotationId) return false;
  if (reviews instanceof Map) return reviews.has(annotationId);
  return Array.from(reviews || []).some((review) => review.annotation_id === annotationId);
}

function normalizeComparisonMark(mark) {
  if (!mark || typeof mark !== "object") return null;
  const expertId = String(mark.expert_id || "anonymous").trim() || "anonymous";
  const groupId = String(mark.group_id || "").trim().toUpperCase();
  const dataset = String(mark.dataset || "").trim();
  const sceneId = String(mark.scene_id || "").trim();
  const firstMethod = String(mark.candidate_a_method || "").trim();
  const secondMethod = String(mark.candidate_b_method || "").trim();
  const pairId = String(
    mark.pair_id || canonicalPairId(dataset, sceneId, firstMethod, secondMethod),
  ).trim();
  const markOrder = Number(mark.mark_order);
  if (!GROUP_IDS.includes(groupId) || !dataset || !sceneId || !firstMethod || !secondMethod ||
      !pairId || !mark.trial_id || !mark.input_path || !mark.candidate_a_path ||
      !mark.candidate_b_path || !Number.isInteger(markOrder) || markOrder < 1) return null;

  return {
    mark_order: markOrder,
    expert_id: expertId,
    group_id: groupId,
    pair_id: pairId,
    trial_id: String(mark.trial_id),
    dataset,
    scene_id: sceneId,
    input_method: String(mark.input_method || "input"),
    candidate_a_method: firstMethod,
    candidate_b_method: secondMethod,
    input_path: String(mark.input_path),
    candidate_a_path: String(mark.candidate_a_path),
    candidate_b_path: String(mark.candidate_b_path),
    marked_at: String(mark.marked_at || "").trim(),
  };
}

function latestComparisonMarks(marks) {
  const latest = new Map();
  for (const mark of marks || []) {
    const normalized = normalizeComparisonMark(mark);
    if (!normalized) continue;
    if (latest.has(normalized.pair_id)) latest.delete(normalized.pair_id);
    latest.set(normalized.pair_id, normalized);
  }
  return latest;
}

function orderedComparisonMarks(marks) {
  const values = marks instanceof Map ? marks.values() : marks || [];
  return Array.from(values).sort((left, right) =>
    Number(left.mark_order) - Number(right.mark_order));
}

function comparisonMarkToCsvRow(mark) {
  return COMPARISON_MARK_HEADER.map((field) => csvEscape(mark?.[field])).join(",");
}

function serializeComparisonMarks(marks) {
  const rows = orderedComparisonMarks(marks).map((mark) => comparisonMarkToCsvRow(mark));
  return `${COMPARISON_MARK_HEADER.join(",")}\r\n${rows.length ? `${rows.join("\r\n")}\r\n` : ""}`;
}

function isComparisonMarked(marks, pairId) {
  if (!pairId) return false;
  if (marks instanceof Map) return marks.has(pairId);
  return Array.from(marks || []).some((mark) => mark.pair_id === pairId);
}

function comparisonMarksToTrials(marks, records) {
  const recordsByPath = new Map(
    (records || []).filter((record) => record?.path).map((record) => [record.path, record]),
  );
  const recordFor = (path, method, itemType) => recordsByPath.get(path) || {
    path,
    method,
    item_type: itemType,
  };
  return orderedComparisonMarks(marks).map((mark) => ({
    pair_id: mark.pair_id,
    trial_id: mark.trial_id,
    dataset: mark.dataset,
    scene_id: mark.scene_id,
    sample_id: mark.scene_id,
    input: recordFor(mark.input_path, mark.input_method, "input"),
    a: recordFor(mark.candidate_a_path, mark.candidate_a_method, "candidate"),
    b: recordFor(mark.candidate_b_path, mark.candidate_b_method, "candidate"),
  }));
}

function timestampValue(result) {
  const timestamp = Date.parse(result?.timestamp || "");
  return Number.isFinite(timestamp) ? timestamp : Number.NEGATIVE_INFINITY;
}

function createGroupCsvRepository(directoryHandle) {
  if (!directoryHandle?.getFileHandle) throw new Error("A writable directory handle is required");
  const cache = new Map();
  let comparisonMarkCache = null;
  const reviewCache = new Map();

  async function writeAll(fileHandle, results) {
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(encoder.encode(serializeResults(results)));
      await writable.close();
    } catch (error) {
      if (typeof writable.abort === "function") await writable.abort().catch(() => {});
      throw error;
    }
  }

  async function appendRow(fileHandle, result) {
    const file = await fileHandle.getFile();
    const writable = await fileHandle.createWritable({ keepExistingData: true });
    try {
      await writable.seek(file.size);
      await writable.write(encoder.encode(`${resultToCsvRow(result)}\r\n`));
      await writable.close();
    } catch (error) {
      if (typeof writable.abort === "function") await writable.abort().catch(() => {});
      throw error;
    }
  }

  async function writeComparisonMarks(fileHandle, marks) {
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(encoder.encode(serializeComparisonMarks(marks)));
      await writable.close();
    } catch (error) {
      if (typeof writable.abort === "function") await writable.abort().catch(() => {});
      throw error;
    }
  }

  async function writeReviews(fileHandle, reviews) {
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(encoder.encode(serializeReviews(reviews)));
      await writable.close();
    } catch (error) {
      if (typeof writable.abort === "function") await writable.abort().catch(() => {});
      throw error;
    }
  }

  async function appendReview(fileHandle, review) {
    const file = await fileHandle.getFile();
    const writable = await fileHandle.createWritable({ keepExistingData: true });
    try {
      await writable.seek(file.size);
      await writable.write(encoder.encode(`${reviewToCsvRow(review)}\r\n`));
      await writable.close();
    } catch (error) {
      if (typeof writable.abort === "function") await writable.abort().catch(() => {});
      throw error;
    }
  }

  async function loadGroup(groupId, allowCreate = true) {
    const filename = groupFilename(groupId);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: allowCreate });
    const file = await fileHandle.getFile();
    const text = await file.text();

    if (!text.trim()) {
      const empty = new Map();
      await writeAll(fileHandle, empty.values());
      cache.set(groupId, empty);
      return { fileHandle, results: empty };
    }

    let document;
    try {
      document = parseCsvDocument(text);
    } catch (error) {
      throw new Error(`Invalid CSV in ${filename}: ${error.message}`);
    }

    const currentHeader = sameHeader(document.header, CSV_HEADER);
    const legacyHeader = sameHeader(document.header, LEGACY_HEADER);
    if (!currentHeader && !legacyHeader) {
      throw new Error(`Invalid CSV header in ${filename}`);
    }

    const normalizedRows = document.rows.map((row) => normalizeResult(row, groupId));
    if (normalizedRows.some((row) => !row)) {
      throw new Error(`Invalid CSV rows in ${filename}`);
    }
    const results = latestResultsByPair(normalizedRows, groupId);
    const needsRewrite = legacyHeader ||
      results.size !== normalizedRows.length ||
      !/[\r\n]$/.test(text);
    if (needsRewrite) {
      await writeAll(fileHandle, results.values());
    }
    cache.set(groupId, results);
    return { fileHandle, results };
  }

  async function readGroup(groupId) {
    const loaded = await loadGroup(groupId);
    return new Map(loaded.results);
  }

  async function loadComparisonMarks() {
    const fileHandle = await directoryHandle.getFileHandle(COMPARISON_MARK_FILENAME, { create: true });
    const file = await fileHandle.getFile();
    const text = await file.text();
    if (!text.trim()) {
      const empty = new Map();
      await writeComparisonMarks(fileHandle, empty.values());
      comparisonMarkCache = empty;
      return { fileHandle, marks: empty };
    }

    let document;
    try {
      document = parseCsvDocument(text);
    } catch (error) {
      throw new Error(`Invalid CSV in ${COMPARISON_MARK_FILENAME}: ${error.message}`);
    }
    if (!sameHeader(document.header, COMPARISON_MARK_HEADER)) {
      throw new Error(`Invalid CSV header in ${COMPARISON_MARK_FILENAME}`);
    }
    const normalizedRows = document.rows.map((row) => normalizeComparisonMark(row));
    if (normalizedRows.some((row) => !row)) {
      throw new Error(`Invalid CSV rows in ${COMPARISON_MARK_FILENAME}`);
    }
    const marks = latestComparisonMarks(normalizedRows);
    if (marks.size !== normalizedRows.length || !/[\r\n]$/.test(text)) {
      await writeComparisonMarks(fileHandle, marks.values());
    }
    comparisonMarkCache = marks;
    return { fileHandle, marks };
  }

  async function readComparisonMarks() {
    const loaded = await loadComparisonMarks();
    return new Map(loaded.marks);
  }

  async function loadReviewGroup(groupId) {
    const filename = reviewGroupFilename(groupId);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const file = await fileHandle.getFile();
    const text = await file.text();
    if (!text.trim()) {
      const empty = new Map();
      await writeReviews(fileHandle, empty.values());
      reviewCache.set(groupId, empty);
      return { fileHandle, reviews: empty };
    }

    let document;
    try {
      document = parseCsvDocument(text);
    } catch (error) {
      throw new Error(`Invalid CSV in ${filename}: ${error.message}`);
    }
    if (!sameHeader(document.header, REVIEW_HEADER)) {
      throw new Error(`Invalid CSV header in ${filename}`);
    }
    const normalizedRows = document.rows.map((row) => normalizeReview(row, groupId));
    if (normalizedRows.some((row) => !row)) {
      throw new Error(`Invalid CSV rows in ${filename}`);
    }
    const reviews = latestReviewsByAnnotation(normalizedRows, groupId);
    if (reviews.size !== normalizedRows.length || !/[\r\n]$/.test(text)) {
      await writeReviews(fileHandle, reviews.values());
    }
    reviewCache.set(groupId, reviews);
    return { fileHandle, reviews };
  }

  async function readReviewGroup(groupId) {
    const loaded = await loadReviewGroup(groupId);
    return new Map(loaded.reviews);
  }

  async function initialize() {
    for (const groupId of GROUP_IDS) {
      await loadGroup(groupId);
      await loadReviewGroup(groupId);
    }
    await loadComparisonMarks();
  }

  async function currentGroup(groupId) {
    if (cache.has(groupId)) return cache.get(groupId);
    return (await loadGroup(groupId)).results;
  }

  async function saveResult(groupId, result) {
    const normalized = normalizeResult(result, groupId);
    if (!normalized) throw new Error("Result is missing canonical pair fields");
    const filename = groupFilename(groupId);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const results = await currentGroup(groupId);
    const exists = results.has(normalized.pair_id);

    if (exists) {
      results.set(normalized.pair_id, normalized);
      await writeAll(fileHandle, results.values());
      return { mode: "rewrite", result: normalized };
    }

    await appendRow(fileHandle, normalized);
    results.set(normalized.pair_id, normalized);
    return { mode: "append", result: normalized };
  }

  async function mergeResults(groupId, incomingResults) {
    const filename = groupFilename(groupId);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const results = await currentGroup(groupId);
    let changed = false;

    for (const incoming of latestResultsByPair(incomingResults, groupId).values()) {
      const existing = results.get(incoming.pair_id);
      if (!existing || timestampValue(incoming) > timestampValue(existing)) {
        results.set(incoming.pair_id, incoming);
        changed = true;
      }
    }

    if (changed) await writeAll(fileHandle, results.values());
    return { changed, results: new Map(results) };
  }

  async function setComparisonMarked(context) {
    const trial = context?.trial;
    const expertId = String(context?.expert_id || "anonymous").trim() || "anonymous";
    const groupId = String(context?.group_id || "").trim().toUpperCase();
    const pairId = String(trial?.pair_id || "").trim();
    if (!GROUP_IDS.includes(groupId) || !pairId) {
      throw new Error("Comparison mark is missing identity fields");
    }

    if (!comparisonMarkCache) await loadComparisonMarks();
    const marks = new Map(comparisonMarkCache);
    const removed = marks.delete(pairId);
    const shouldMark = Boolean(context?.marked);
    if (shouldMark) {
      const nextOrder = orderedComparisonMarks(marks)
        .reduce((maximum, mark) => Math.max(maximum, Number(mark.mark_order) || 0), 0) + 1;
      const mark = normalizeComparisonMark({
        mark_order: nextOrder,
        expert_id: expertId,
        group_id: groupId,
        pair_id: pairId,
        trial_id: trial.trial_id,
        dataset: trial.dataset,
        scene_id: trial.scene_id,
        input_method: trial.input?.method,
        candidate_a_method: trial.a?.method,
        candidate_b_method: trial.b?.method,
        input_path: trial.input?.path,
        candidate_a_path: trial.a?.path,
        candidate_b_path: trial.b?.path,
        marked_at: context.marked_at || new Date().toISOString(),
      });
      if (!mark) throw new Error("Comparison is missing reconstruction fields");
      marks.set(mark.pair_id, mark);
    }

    const fileHandle = await directoryHandle.getFileHandle(COMPARISON_MARK_FILENAME, { create: true });
    await writeComparisonMarks(fileHandle, marks.values());
    comparisonMarkCache = marks;
    return {
      marked: shouldMark,
      affectedRows: shouldMark || removed ? 1 : 0,
      marks: new Map(marks),
    };
  }

  async function setReviewed(context) {
    const reviewGroupId = String(context?.review_group_id || "").trim().toUpperCase();
    const reviewerId = String(context?.reviewer_id || "anonymous").trim() || "anonymous";
    const sourceResult = normalizeResult(context?.source_result, context?.source_result?.group_id);
    const annotationId = annotationReviewId(sourceResult);
    if (!GROUP_IDS.includes(reviewGroupId) || !sourceResult || !annotationId) {
      throw new Error("Review is missing annotation identity fields");
    }

    if (!reviewCache.has(reviewGroupId)) await loadReviewGroup(reviewGroupId);
    const current = reviewCache.get(reviewGroupId);
    const reviews = new Map(current);
    const removed = reviews.delete(annotationId);
    const shouldReview = Boolean(context?.reviewed);
    const filename = reviewGroupFilename(reviewGroupId);
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });

    if (shouldReview) {
      const reviewResult = normalizeResult(context?.review_result || sourceResult, sourceResult.group_id);
      const choices = {
        input_vs_a: reviewResult?.input_vs_a,
        input_vs_b: reviewResult?.input_vs_b,
        a_vs_b: reviewResult?.a_vs_b,
      };
      const ranking = summarizeRanking(choices);
      const validChoices = new Set(["left", "right", "similar"]);
      if (!reviewResult || Object.values(choices).some((value) => !validChoices.has(value)) ||
          ranking.hasConflict) {
        throw new Error("Reviewed choices must form a complete, consistent ranking");
      }
      const nextOrder = orderedReviews(reviews)
        .reduce((maximum, review) => Math.max(maximum, Number(review.review_order) || 0), 0) + 1;
      const review = normalizeReview({
        review_order: nextOrder,
        review_group_id: reviewGroupId,
        reviewer_id: reviewerId,
        source_group_id: sourceResult.group_id,
        source_expert_id: sourceResult.expert_id,
        annotation_id: annotationId,
        pair_id: sourceResult.pair_id,
        trial_id: sourceResult.trial_id,
        dataset: sourceResult.dataset,
        scene_id: sourceResult.scene_id,
        input_method: sourceResult.input_method,
        candidate_a_method: sourceResult.candidate_a_method,
        candidate_b_method: sourceResult.candidate_b_method,
        input_path: sourceResult.input_path,
        candidate_a_path: sourceResult.candidate_a_path,
        candidate_b_path: sourceResult.candidate_b_path,
        input_vs_a: choices.input_vs_a,
        input_vs_b: choices.input_vs_b,
        a_vs_b: choices.a_vs_b,
        ranking: ranking.text,
        source_elapsed_ms: sourceResult.elapsed_ms,
        source_timestamp: sourceResult.timestamp,
        reviewed_at: context.reviewed_at || new Date().toISOString(),
      }, reviewGroupId);
      if (!review) throw new Error("Review is missing source annotation fields");
      reviews.set(annotationId, review);
      if (removed) {
        await writeReviews(fileHandle, reviews.values());
      } else {
        await appendReview(fileHandle, review);
      }
    } else if (removed) {
      await writeReviews(fileHandle, reviews.values());
    }

    reviewCache.set(reviewGroupId, reviews);
    return {
      reviewed: shouldReview,
      affectedRows: shouldReview || removed ? 1 : 0,
      reviews: new Map(reviews),
    };
  }

  return {
    directoryHandle,
    initialize,
    mergeResults,
    readGroup,
    readComparisonMarks,
    readReviewGroup,
    saveResult,
    setComparisonMarked,
    setReviewed,
  };
}

const MosGroupCsv = {
  GROUP_IDS,
  COMPARISON_MARK_FILENAME,
  COMPARISON_MARK_HEADER,
  REVIEW_HEADER,
  annotationReviewId,
  comparisonMarksToTrials,
  createGroupCsvRepository,
  groupFilename,
  isAnnotationReviewed,
  isComparisonMarked,
  latestComparisonMarks,
  latestReviewsByAnnotation,
  latestResultsByPair,
  normalizeComparisonMark,
  normalizeReview,
  normalizeResult,
  orderedComparisonMarks,
  orderedReviews,
  parseCsv,
  reviewGroupFilename,
  selectUnfinishedTrials,
  serializeComparisonMarks,
  serializeReviews,
  serializeResults,
};

if (root) root.MosGroupCsv = MosGroupCsv;
if (typeof module !== "undefined") module.exports = MosGroupCsv;
})(typeof window !== "undefined" ? window : globalThis);
