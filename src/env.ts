/** Environment access with one failure mode: a missing key stops the run at the
 *  top rather than halfway through a paid embedding job. */

function read(name: string): string | undefined {
  const v = process.env[name];
  return v && v.trim().length > 0 ? v.trim() : undefined;
}

export function required(name: string): string {
  const v = read(name);
  if (!v) throw new Error(`Missing ${name}. Copy .env.example to .env and fill it in.`);
  return v;
}

export function optional(name: string): string | undefined {
  return read(name);
}

export const env = {
  get databaseUrl() { return required("DATABASE_URL"); },
  get typesafeKey() { return required("TYPESAFE_API_KEY"); },
  get azureKey() { return required("AZURE_API_KEY"); },
  get azureResource() { return required("AZURE_RESOURCE_NAME"); },
  get azureEmbeddingDeployment() { return process.env.AZURE_EMBEDDING_DEPLOYMENT ?? "text-embedding-3-large"; },
  get azureEmbeddingApiVersion() { return process.env.AZURE_EMBEDDING_API_VERSION ?? "2024-02-01"; },
  get voyageKey() { return optional("VOYAGE_API_KEY"); },
  get corpusSourceDir() { return required("CORPUS_SOURCE_DIR"); },
  get groundTruthCsv() { return required("GROUND_TRUTH_CSV"); },
};
