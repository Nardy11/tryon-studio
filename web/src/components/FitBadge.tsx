import { fitText, fitTone, lengthText, type Fit } from '../fit';

/** "Too small · It only goes up to L…": the verdict in words, colour only as a second signal. */
export function FitBadge({ fit, stock, compact = false }: { fit: Fit; stock: number[]; compact?: boolean }) {
  const { title, detail } = fitText(fit, stock);
  if (!title) return null;
  const length = lengthText(fit.length);
  // Width fits but length doesn't: still worth a warning colour.
  const tone = fit.verdict === 'fits' && length ? 'warn' : fitTone(fit.verdict);
  return (
    <p className={`fit-badge ${tone} ${compact ? 'compact' : ''}`} role="status">
      <strong>{length && fit.verdict === 'fits' ? 'Fits in width' : title}</strong>
      <span>{detail}</span>
      {length && <span className="fit-length">{length}</span>}
    </p>
  );
}
