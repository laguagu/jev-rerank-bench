import { readFileSync } from "node:fs";
import { parseCsv } from "./csv";

export interface EvalQuery {
  id: string;
  /** The original `node` cell; not unique across rows. */
  node: string;
  question: string;
  /** Ground-truth document file names. */
  files: string[];
  /** Ground-truth section headings, when the row names them. */
  sections: string[];
  category: "TES" | "LAW" | "GUIDE" | string;
  hopType: string;
  /** The verified answer. Not used for retrieval scoring; kept for inspection. */
  answer: string;
}

/**
 * The `files` column is comma-separated, but several of these documents have a
 * comma in the file name ("… _LS 2025–2028, voimassa 1.5.2025 lukien _ ….md").
 * Splitting naively loses 22 of 102 references. Rejoin fragments until each
 * piece ends in `.md`.
 */
export function splitFileList(raw: string): string[] {
  const parts = raw.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  const out: string[] = [];
  let buf = "";
  for (const part of parts) {
    buf = buf.length > 0 ? `${buf}, ${part}` : part;
    if (buf.toLowerCase().endsWith(".md")) {
      out.push(buf);
      buf = "";
    }
  }
  if (buf.length > 0) out.push(buf);
  return out;
}

export function loadGroundTruth(csvPath: string): EvalQuery[] {
  const rows = parseCsv(readFileSync(csvPath, "utf8"), ";");
  // `node` repeats across rows: the same agreement supplies two questions. Keep
  // the node visible but make the id unique, or per-query results collide.
  const seen = new Map<string, number>();
  return rows
    .filter((r) => (r.question ?? "").length > 0)
    .map((r) => {
      const node = r.node || "Q";
      const n = (seen.get(node) ?? 0) + 1;
      seen.set(node, n);
      return { node, id: n === 1 ? node : `${node}.${n}` , row: r };
    })
    .map(({ node, id, row: r }) => ({
      id,
      node,
      question: r.question!,
      files: splitFileList(r.files ?? ""),
      sections: (r.source_sections ?? "")
        .split(/[;|]|\s*,\s*(?=\d+[.\s]*§)/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
      category: r.category ?? "",
      hopType: r.hop_type ?? "",
      answer: r.answer ?? "",
    }));
}

/**
 * Reduce a heading to the words that identify it.
 *
 * The ground truth and the documents disagree on everything except the words:
 * "§ 36 Ammattiyhdistysjäsenmaksun pidättäminen" against "36 § Ammattiyhdistys…",
 * "Luku 1.1 Palkkakäsite" against "Palkkakäsite", and
 * "KVTES 2025-2028 / III luku Työaika / 33 § Säännöllisen…" against just the last
 * segment. Matching on the raw string scores retrieval on a formatting
 * convention, so the numbering, the section marks and the structural words are
 * dropped and only the content words are compared.
 */
const STRUCTURAL = new Set(["luku", "mom", "momentti", "pykälä", "liite", "osa", "kohta", "ja", "tai"]);
const ROMAN = /^[ivxlcdm]+$/;

export function headingTokens(h: string): Set<string> {
  const lastSegment = h.includes("/") ? h.slice(h.lastIndexOf("/") + 1) : h;
  return new Set(
    lastSegment
      .toLowerCase()
      .replace(/[§()[\]{}.:,;"'–—-]/g, " ")
      .split(/\s+/)
      .map((t) => t.trim())
      .filter((t) => t.length > 2 && !/^\d+$/.test(t) && !STRUCTURAL.has(t) && !ROMAN.test(t)),
  );
}

/** The pykälä number, written either before or after the section mark. */
export function sectionNumber(h: string): number | null {
  const m = /(\d+)\s*\.?\s*§|§\s*(\d+)/.exec(h);
  if (!m) return null;
  const n = Number(m[1] ?? m[2]);
  return Number.isFinite(n) ? n : null;
}

/** Kept for the raw-string containment check that still catches long headings. */
export function normaliseHeading(h: string): string {
  return h
    .toLowerCase()
    .replace(/[§.:,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * True when the two headings name the same section.
 *
 * One side being a subset of the other is the common case — the ground truth
 * adds "Luku 1.1", the document adds a trailing qualifier — so overlap is
 * measured against the smaller set, with two content words required so that a
 * single shared word like "työaika" does not match every section in the file.
 */
export function headingsMatch(gold: string, candidate: string): boolean {
  // Dropping the numbering makes "Luku 1.1 Palkkakäsite" match "Palkkakäsite",
  // but it also makes "32. § Ikääntyneiden työntekijöiden työajan lyhentäminen"
  // match "37. §" of the same name — these agreements repeat a section under
  // two numbers for two employee groups. When both sides carry a section
  // number, it has to agree.
  const ga = sectionNumber(gold);
  const gb = sectionNumber(candidate);
  if (ga !== null && gb !== null && ga !== gb) return false;

  const a = headingTokens(gold);
  const b = headingTokens(candidate);
  if (a.size === 0 || b.size === 0) return false;
  let shared = 0;
  for (const t of a) if (b.has(t)) shared++;
  const smaller = Math.min(a.size, b.size);
  if (shared === smaller && smaller >= 1) return true;
  return shared >= 2 && shared / smaller >= 0.6;
}
