/**
 * Which corpus a run works against.
 *
 * The two live in separate Postgres schemas so both stay queryable at once and
 * nothing has to be reloaded to switch. Neither uses `public`: an unqualified
 * name in a `DROP` statement falls through to it, which is how loading the
 * second dataset once deleted the first one's tables.
 *
 * Selected with `--dataset=mupler` on any CLI, or `DATASET=mupler`.
 */
export interface DatasetConfig {
  name: string;
  /** Where `corpus.json` and the embedding cache live. */
  dir: string;
  /** Postgres schema holding `documents` and `chunks`. */
  schema: string;
  /** Whether the ground truth names a section inside the document. */
  hasSectionGroundTruth: boolean;
  /** Shown in reports. */
  label: string;
  /** Whether the corpus may be redistributed with this repository. */
  publishable: boolean;
}

const DATASETS: Record<string, DatasetConfig> = {
  "fi-tes": {
    name: "fi-tes",
    dir: "data/fi-tes",
    schema: "fi_tes",
    hasSectionGroundTruth: true,
    label: "Finnish collective agreements and statutes (not redistributable)",
    publishable: false,
  },
  mupler: {
    name: "mupler",
    dir: "data/mupler",
    schema: "mupler",
    hasSectionGroundTruth: false,
    label: "MuPLeR-fi legal retrieval (EUPL-1.2)",
    publishable: true,
  },
};

export function activeDataset(): DatasetConfig {
  const arg = process.argv.find((a) => a.startsWith("--dataset="))?.split("=")[1];
  const name = arg ?? process.env.DATASET ?? "fi-tes";
  const ds = DATASETS[name];
  if (!ds) throw new Error(`unknown dataset "${name}". Known: ${Object.keys(DATASETS).join(", ")}`);
  return ds;
}

export const DATASET = activeDataset();
