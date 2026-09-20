/**
 * Checks that the ground truth is reachable at all before any metric is
 * believed.
 *
 * Document level is easy — `prepare-corpus` already guarantees the file is
 * indexed. Section level is not: the ground truth writes "32. § Ikääntyneiden
 * työntekijöiden työajan lyhentäminen" and the document may write it
 * differently, in which case `isRelevant` can never return true and the query
 * scores zero for every strategy alike. That is a measurement bug, not a
 * retrieval result, so it has to be counted and named.
 */
import { readFileSync } from "node:fs";
import type { Chunk } from "../corpus";
import type { EvalQuery } from "../dataset";
import { DATASET } from "../dataset-config";
import { isRelevant } from "../metrics";
import type { Candidate } from "../search";

const corpus = JSON.parse(readFileSync(`${DATASET.dir}/corpus.json`, "utf8")) as { chunks: Chunk[]; queries: EvalQuery[] };

const byDoc = new Map<string, Chunk[]>();
for (const c of corpus.chunks) {
  const list = byDoc.get(c.docFile);
  if (list) list.push(c);
  else byDoc.set(c.docFile, [c]);
}

const asCandidate = (c: Chunk): Candidate => ({
  chunkId: 0,
  docFile: c.docFile,
  docTitle: c.docTitle,
  heading: c.heading,
  headingPath: c.headingPath,
  body: c.text,
  score: 0,
});

let docOk = 0;
let sectionOk = 0;
const unreachable: { id: string; sections: string[]; nearest: string[] }[] = [];

for (const q of corpus.queries) {
  const pool = q.files.flatMap((f) => byDoc.get(f) ?? []);
  if (pool.length > 0) docOk++;
  const matches = pool.filter((c) => isRelevant(asCandidate(c), q, "section"));
  if (matches.length > 0) sectionOk++;
  else {
    const want = q.sections[0] ?? "";
    const firstWord = want.replace(/^\d+[.\s§]*/, "").split(/\s+/)[0]?.toLowerCase() ?? "";
    const nearest = pool
      .map((c) => c.heading)
      .filter((h) => firstWord.length > 4 && h.toLowerCase().includes(firstWord))
      .slice(0, 3);
    unreachable.push({ id: q.id, sections: q.sections, nearest });
  }
}

const n = corpus.queries.length;
console.log(`queries                      ${n}`);
console.log(`gold document indexed        ${docOk}/${n}`);
console.log(`gold section matchable       ${sectionOk}/${n}`);
console.log(`chunks per gold document     ${Math.round(corpus.chunks.length / byDoc.size)} avg\n`);

if (unreachable.length > 0) {
  console.log(`${unreachable.length} queries whose section heading never matches a chunk — these score 0`);
  console.log(`at section level for every strategy, so they measure the heading matcher, not retrieval:\n`);
  for (const u of unreachable.slice(0, 15)) {
    console.log(`  ${u.id.padEnd(10)} wants "${u.sections.join(" | ").slice(0, 70)}"`);
    if (u.nearest.length > 0) console.log(`  ${" ".repeat(10)} nearest in document: ${u.nearest.map((h) => `"${h.slice(0, 50)}"`).join(", ")}`);
  }
  if (unreachable.length > 15) console.log(`  … and ${unreachable.length - 15} more`);
}
