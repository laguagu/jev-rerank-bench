import { DATASET } from "../dataset-config";
import type { Candidate } from "../search";

/**
 * The relevance judgment, per corpus.
 *
 * This file exists because of a measured failure. The first version had one
 * question, written for the Finnish collective-agreement corpus, which requires the model to check
 * *provenance*: hundreds of Finnish collective agreements repeat each other
 * section for section, so the passage text alone does not identify the answer
 * and the criteria say so explicitly — "it comes from the document the question
 * is about: the sector, agreement, statute or guideline the question names".
 *
 * Run against MuPLeR, that question scored **below no reranking at all**
 * (66.5% R@1 against 73.0%). MuPLeR's passages carry a bare numeric id as their
 * title, so "comes from the document the question names" is not a condition the
 * model can ever verify — and a Noul that cannot verify its condition answers
 * no. The question was wrong, not the model. The state was wrong too: sending a
 * `document: "6099"` field adds a distractor and nothing else.
 *
 * So each corpus gets a judgment written for what actually distinguishes a
 * right answer there, and the state carries only the fields that judgment uses.
 * Both are shared with the LLM baselines by import, so the comparison stays a
 * comparison of models.
 */

export interface RelevanceSpec {
  instructions: Record<string, string>;
  criteria: { true: string; false: string };
  /** Four ordered situations, worst first. Numerals stay out of the text. */
  rubric: readonly [string, string, ...string[]];
  /** Only the fields the judgment uses. */
  state: (c: Candidate) => Record<string, string>;
}

/**
 * Finnish collective agreements: the passage text is often identical across
 * documents, and only the source document separates a right answer from a wrong
 * one.
 */
const FI_TES: RelevanceSpec = {
  instructions: {
    task:
      "A user asked a question in Finnish about employment law, a collective agreement (tyoehtosopimus) " +
      "or an official guideline. Decide whether this one passage is a source that answers that question.",
    corpus:
      "The passages come from Finnish statutes, official guidance and collective agreements. Collective " +
      "agreements repeat one another section by section, so the same provision appears in many documents " +
      "for different sectors, employers and years.",
  },
  criteria: {
    true:
      "The passage states the specific provision, figure, table row or rule the question asks for, and it " +
      "comes from the document the question is about: the sector, agreement, statute or guideline the " +
      "question names or clearly implies. Someone reading this passage alone could answer the question.",
    false:
      "The passage is only on a related topic, or it states the same kind of rule for a different sector, " +
      "agreement, statute or year than the question asks about, or it names the subject without giving the " +
      "provision the question asks for.",
  },
  rubric: [
    "The passage is unrelated to what the question asks about.",
    "The passage is about the same general subject but from a different document, sector, agreement or year than the question asks about, or it only mentions the subject in passing.",
    "The passage is from the document the question is about and covers the right subject, but it does not state the specific provision, figure or rule the question asks for.",
    "The passage is from the document the question is about and states the exact provision, figure, table row or rule the question asks for.",
  ],
  state: (c) => ({ document: c.docTitle, section: c.headingPath || c.heading, text: c.body }),
};

/**
 * MuPLeR: passages are self-contained and their ids carry no meaning, so the
 * judgment is whether this passage contains what the question asks about.
 * Nothing is asked about provenance, because nothing in the state could answer
 * it.
 */
const PASSAGE: RelevanceSpec = {
  instructions: {
    task: "A user asked a question. Decide whether this passage is a source that answers it.",
    corpus:
      "The passages come from legal and policy documents. The question was written from one particular " +
      "passage and may paraphrase it, generalise it, or ask about its subject in different words, so the " +
      "wording will often not match.",
  },
  criteria: {
    true:
      "The passage contains the specific statement, rule, finding or reasoning the question asks about, so " +
      "that someone reading this passage alone could answer the question.",
    false:
      "The passage is on a related subject, or shares vocabulary with the question, but does not contain the " +
      "specific statement the question asks about.",
  },
  rubric: [
    "The passage is about a different subject from the one the question asks about.",
    "The passage shares vocabulary or a general topic with the question but does not address what it asks.",
    "The passage addresses what the question asks about but stops short of the specific statement, rule or finding the question is after.",
    "The passage contains the specific statement, rule, finding or reasoning the question asks about, so the question can be answered from this passage alone.",
  ],
  state: (c) => ({ text: c.body }),
};

const SPECS: Record<string, RelevanceSpec> = { "fi-tes": FI_TES, mupler: PASSAGE };

/** `--question=passage` forces the portable judgment onto any corpus, which is
 *  how the "was the question or the model at fault" comparison is run. */
export function relevanceSpec(): RelevanceSpec {
  const forced = process.argv.find((a) => a.startsWith("--question="))?.split("=")[1];
  if (forced) {
    const spec = forced === "passage" ? PASSAGE : forced === "fi-tes" ? FI_TES : SPECS[forced];
    if (!spec) throw new Error(`unknown --question=${forced}. Known: passage, fi-tes`);
    return spec;
  }
  return SPECS[DATASET.name] ?? PASSAGE;
}

export const SPEC = relevanceSpec();
