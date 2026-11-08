import { fitText, fitTone, type Fit } from '../fit';

/** "Too small · It only goes up to L…": the verdict in words, colour only as a second signal. */
export function FitBadge({ fit, stock, compact = false }: { fit: Fit; stock: number[]; compact?: boolean }) {
  const { title, detail } = fitText(fit, stock);
  if (!title) return null;
  return (
    <p className={`fit-badge ${fitTone(fit.verdict)} ${compact ? 'compact' : ''}`} role="status">
      <strong>{title}</strong>
      <span>{detail}</span>
    </p>
  );
}
