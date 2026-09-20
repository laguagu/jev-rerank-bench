-- jev-search-lab schema. One table of chunks carrying both retrieval
-- representations, so FTS, vector and hybrid all read the same rows and a
-- difference in the metrics can only come from the ranking.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS {{schema}};

-- Every name here is schema-qualified, and deliberately so.
--
-- This file used to `SET search_path = {{schema}}, public` and then
-- `DROP TABLE IF EXISTS chunks CASCADE`. On the first load of a second dataset
-- the target schema has no `chunks` yet, so that name resolves to the *next*
-- entry on the search path — and the drop takes out the first dataset's table
-- instead. Loading MuPLeR silently deleted the private corpus that way, and
-- nothing noticed for half an hour because every run in between was reading
-- cached shortlists rather than the database. DDL does not get to rely on a
-- search path.
DROP TABLE IF EXISTS {{schema}}.chunks CASCADE;
DROP TABLE IF EXISTS {{schema}}.documents CASCADE;

CREATE TABLE {{schema}}.documents (
  file    text PRIMARY KEY,
  title   text NOT NULL,
  folder  text NOT NULL,
  bytes   bigint NOT NULL
);

CREATE TABLE {{schema}}.chunks (
  id            bigserial PRIMARY KEY,
  doc_file      text NOT NULL REFERENCES {{schema}}.documents(file) ON DELETE CASCADE,
  ordinal       int  NOT NULL,
  heading       text NOT NULL DEFAULT '',
  heading_path  text NOT NULL DEFAULT '',
  body          text NOT NULL,
  embedding     vector(1536),
  -- `finnish` is Postgres' built-in snowball configuration. `unaccent` is not
  -- applied: Finnish ä and ö are distinct letters, not accents, and folding
  -- them merges words the corpus actually distinguishes.
  tsv           tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('finnish', coalesce(heading_path, '')), 'A') ||
                  setweight(to_tsvector('finnish', coalesce(body, '')), 'B')
                ) STORED,
  UNIQUE (doc_file, ordinal)
);

CREATE INDEX chunks_tsv_idx ON {{schema}}.chunks USING gin (tsv);
CREATE INDEX chunks_doc_idx ON {{schema}}.chunks (doc_file);
