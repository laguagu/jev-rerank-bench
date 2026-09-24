import postgres from "postgres";
import { CONFIG } from "./config";
import { DATASET } from "./dataset-config";
import { env } from "./env";

/**
 * Connects to the *unpooled* endpoint.
 *
 * `hnsw.ef_search` has to be raised: pgvector defaults it to 40, and the hybrid
 * query fetches 60 dense candidates before fusion, so the default silently caps
 * the dense side below the depth the benchmark asks for. Neon refuses
 * `ALTER DATABASE … SET hnsw.ef_search` and its pooler rejects the parameter in
 * the startup packet ("unsupported startup parameter in options"), so the only
 * place it can be set once per connection is a direct endpoint. A batch
 * benchmark has no use for transaction pooling anyway, and the direct endpoint
 * also allows prepared statements.
 *
 * `search_path` points at the active dataset's schema, so every query in
 * `search.ts` names `chunks` and `documents` unqualified and still reads the
 * right corpus.
 */
const directUrl = env.databaseUrl.replace("-pooler.", ".");

export const sql = postgres(directUrl, {
  max: 8,
  idle_timeout: 20,
  connect_timeout: 30,
  connection: { options: `-c hnsw.ef_search=${CONFIG.hnswEfSearch} -c search_path=${DATASET.schema},public` },
  onnotice: () => {},
});

export async function close(): Promise<void> {
  await sql.end({ timeout: 5 });
}
