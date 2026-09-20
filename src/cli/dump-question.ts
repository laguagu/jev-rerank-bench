/**
 * Write the active corpus's relevance judgment to JSON.
 *
 * The Laya baseline runs in Python, so it cannot import `questions.ts`. Dumping
 * the spec keeps the guarantee that every model is asked the identical
 * question: the wording has exactly one source, and the Python side reads it
 * rather than restating it.
 */
import { writeFileSync } from "node:fs";
import { DATASET } from "../dataset-config";
import { SPEC } from "../rerank/questions";

const out = `${DATASET.dir}/question-spec.json`;
writeFileSync(
  out,
  JSON.stringify({ dataset: DATASET.name, instructions: SPEC.instructions, criteria: SPEC.criteria, rubric: SPEC.rubric }, null, 2),
  "utf8",
);
console.log(`wrote ${out}`);
console.log(`  task: ${SPEC.instructions.task}`);
