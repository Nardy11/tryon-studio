import { SIZES } from '../fit';

/** "Sizes in stock" for one outfit: XS S M L XL, any combination. None = no size check. */
export function SizeChips({ value, onChange }: { value: number[]; onChange: (v: number[]) => void }) {
  const toggle = (i: number) => onChange(value.includes(i) ? value.filter((x) => x !== i) : [...value, i].sort((a, b) => a - b));
  return (
    <div className="field">
      <span className="field-label" id="sizes-label">
        Sizes in stock
      </span>
      <div className="size-chips" role="group" aria-labelledby="sizes-label">
        {SIZES.map((s, i) => (
          <button key={s} type="button" aria-pressed={value.includes(i)} className={value.includes(i) ? 'on' : ''} onClick={() => toggle(i)}>
            {s}
          </button>
        ))}
      </div>
      {!value.length && <p className="pieces-note">Pick the sizes this outfit comes in to check the fit.</p>}
    </div>
  );
}
