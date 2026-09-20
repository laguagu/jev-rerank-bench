-- Built after the vectors are loaded: HNSW on an empty table is wasted work and
-- pgvector builds faster over existing rows.
--
-- 768MB, not the 2GB pgvector's own guidance suggests. 37 440 vectors at 1536
-- dimensions are 230MB of raw data and the graph a few times that, but the
-- default Neon compute for this project is 1 CU. Asking for more than the node
-- has does not make the build faster; it makes it a build that gets killed.
SET maintenance_work_mem = '768MB';

-- Schema-qualified, like everything in schema.sql. See the note there.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON {{schema}}.chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

ANALYZE {{schema}}.chunks;
