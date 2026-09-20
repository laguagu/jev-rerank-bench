/**
 * Does the vector query actually use the HNSW index?
 *
 * Run after an ingest, against whichever dataset `--dataset` selects. It prints
 * the chosen scan node and the execution time for four shapes of the same
 * top-60 nearest-neighbour query, so an index that is silently not being used
 * shows up as a Seq Scan rather than as a number nobody questions.
 */
import { close, sql } from "../src/db";

const [row] = await sql<{ v: string }[]>`SELECT embedding::text AS v FROM chunks ORDER BY id LIMIT 1`;
const v = row!.v;
const [ef] = await sql<{ hnsw_ef_search: string }[]>`SHOW hnsw.ef_search`;
console.log(`hnsw.ef_search = ${Object.values(ef!)[0]}\n`);

const cases: { label: string; query: string; args: unknown[] }[] = [
  {
    label: "plain, parameterised",
    query: "SELECT id FROM chunks ORDER BY embedding <=> $1::vector LIMIT 60",
    args: [v],
  },
  {
    label: "plain, inlined literal",
    query: `SELECT id FROM chunks ORDER BY embedding <=> '${v}'::vector LIMIT 60`,
    args: [],
  },
  {
    label: "window inside LIMIT",
    query:
      "SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank " +
      "FROM chunks ORDER BY embedding <=> $1::vector LIMIT 60",
    args: [v],
  },
  {
    label: "window outside LIMIT",
    query:
      "SELECT id, ROW_NUMBER() OVER () AS rank FROM " +
      "(SELECT id FROM chunks ORDER BY embedding <=> $1::vector LIMIT 60) t",
    args: [v],
  },
];

console.log(`${"shape".padEnd(24)} ${"time".padStart(11)}   scan node`);
for (const c of cases) {
  const plan = await sql.unsafe(`EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) ${c.query}`, c.args as never[]);
  const text = plan.map((r) => (r as Record<string, string>)["QUERY PLAN"]).join("\n");
  const scan = text.split("\n").map((s) => s.trim()).find((l) => /Index Scan|Seq Scan/.test(l)) ?? "?";
  const ms = /Execution Time: ([\d.]+)/.exec(text)?.[1] ?? "?";
  console.log(`${c.label.padEnd(24)} ${`${ms} ms`.padStart(11)}   ${scan.slice(0, 66)}`);
}

await close();
