import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, join } from "node:path";
import { CONFIG } from "./config";

export interface SourceDoc {
  /** File name as it appears in the ground truth, e.g. `TES12_Example2501.md`. */
  file: string;
  /** `collective-agreements` | `guidelines` | `laws` | `other` */
  folder: string;
  title: string;
  path: string;
  bytes: number;
}

export interface Chunk {
  docFile: string;
  docTitle: string;
  folder: string;
  /** Position of the chunk within its document, from zero. */
  ordinal: number;
  /** The nearest enclosing markdown heading, verbatim, or "" at the top of a file. */
  heading: string;
  /** Every enclosing heading, outermost first, joined with " > ". */
  headingPath: string;
  text: string;
}

/** Housekeeping files that sit next to the corpus but are not corpus. */
const NOT_CORPUS = new Set(["links.md", "readme.md", "index.md", "manifest.md"]);

/**
 * The ground truth identifies a document by bare file name, so file name is the
 * primary key throughout. Two folders each hold a `LINKS.md`, which is both not
 * a document and a duplicate key — indexing it produced two `documents` rows
 * with the same name and the load failed on the unique constraint after the
 * embeddings had already been paid for. Both problems are filtered here rather
 * than worked around at insert time, so the corpus and the database agree.
 */
export function listDocuments(sourceDir: string): SourceDoc[] {
  const out: SourceDoc[] = [];
  const seen = new Set<string>();
  const skipped: string[] = [];

  const walk = (dir: string, folder: string) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry);
      const st = statSync(p);
      if (st.isDirectory()) {
        walk(p, entry);
        continue;
      }
      if (!entry.endsWith(".md")) continue;
      if (NOT_CORPUS.has(entry.toLowerCase())) continue;
      if (seen.has(entry)) {
        skipped.push(`${folder}/${entry}`);
        continue;
      }
      seen.add(entry);
      out.push({
        file: entry,
        folder,
        title: entry.replace(/\.md$/, "").replace(/^[A-Z]+\d+_/, ""),
        path: p,
        bytes: st.size,
      });
    }
  };
  walk(sourceDir, basename(sourceDir));
  if (skipped.length > 0) console.warn(`listDocuments: skipped ${skipped.length} duplicate file name(s): ${skipped.join(", ")}`);
  return out.sort((a, b) => a.file.localeCompare(b.file, "fi"));
}

/** Deterministic PRNG so a rerun selects the same distractors. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Split a markdown document into retrieval chunks.
 *
 * Sections come first: these documents are legal texts where a `§` heading is
 * the unit a reader cites, and the ground truth is expressed as a heading. Long
 * sections are then windowed with overlap, and each window keeps the heading so
 * a fragment of a pay table still carries the section it belongs to.
 */
export function chunkDocument(doc: SourceDoc, raw: string): Chunk[] {
  const lines = raw.split(/\r?\n/);
  interface Section { headingPath: string; heading: string; body: string[] }
  const sections: Section[] = [];
  const stack: { level: number; text: string }[] = [];
  let current: Section = { headingPath: "", heading: "", body: [] };

  const pushCurrent = () => {
    if (current.body.join("\n").trim().length > 0 || current.heading) sections.push(current);
  };

  for (const line of lines) {
    const m = /^(#{1,4})\s+(.*)$/.exec(line);
    if (m) {
      pushCurrent();
      const level = m[1]!.length;
      const text = m[2]!.trim();
      while (stack.length > 0 && stack[stack.length - 1]!.level >= level) stack.pop();
      stack.push({ level, text });
      current = {
        heading: text,
        headingPath: stack.map((s) => s.text).join(" > "),
        body: [],
      };
    } else current.body.push(line);
  }
  pushCurrent();

  const chunks: Chunk[] = [];
  let ordinal = 0;
  let carry: Section | null = null;

  for (const section of sections) {
    let merged = section;
    if (carry) {
      merged = {
        heading: carry.heading || section.heading,
        headingPath: carry.headingPath || section.headingPath,
        body: [...carry.body, carry.heading ? `## ${section.heading}` : "", ...section.body],
      };
      carry = null;
    }
    const body = merged.body.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    // A heading with almost no text under it carries no signal on its own.
    if (body.length < CONFIG.chunkMinChars && section !== sections[sections.length - 1]) {
      carry = merged;
      continue;
    }
    for (const window of windowText(body, CONFIG.chunkMaxChars, CONFIG.chunkOverlapChars)) {
      chunks.push({
        docFile: doc.file,
        docTitle: doc.title,
        folder: doc.folder,
        ordinal: ordinal++,
        heading: merged.heading,
        headingPath: merged.headingPath,
        text: window,
      });
    }
  }
  return chunks;
}

/** Split on paragraph boundaries where possible, hard-cut only when a single
 *  paragraph (a wide pay table, usually) exceeds the window on its own. */
function windowText(text: string, maxChars: number, overlap: number): string[] {
  if (text.length <= maxChars) return text.length > 0 ? [text] : [];
  const paragraphs = text.split(/\n{2,}/);
  const out: string[] = [];
  let buf = "";
  const flush = () => {
    if (buf.trim().length > 0) out.push(buf.trim());
    buf = "";
  };
  for (const p of paragraphs) {
    if (p.length > maxChars) {
      flush();
      for (let i = 0; i < p.length; i += maxChars - overlap) out.push(p.slice(i, i + maxChars));
      continue;
    }
    if (buf.length + p.length + 2 > maxChars) {
      const tail = buf.slice(Math.max(0, buf.length - overlap));
      flush();
      buf = tail.length > 0 && tail.length < p.length ? `${tail}\n\n${p}` : p;
    } else buf = buf.length > 0 ? `${buf}\n\n${p}` : p;
  }
  flush();
  return out;
}

/** What actually goes to the embedding model. The heading path is prepended
 *  because a windowed fragment of a pay table is unidentifiable without it. */
export function embeddingInput(chunk: Chunk): string {
  const head = [chunk.docTitle, chunk.headingPath].filter(Boolean).join(" — ");
  return head ? `${head}\n\n${chunk.text}` : chunk.text;
}

export function readDoc(doc: SourceDoc): string {
  return readFileSync(doc.path, "utf8");
}
