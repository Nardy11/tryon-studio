import { useState } from 'react';

export type OutfitThumb = { id: string; url: string; label: string };

type Props = {
  outfits: OutfitThumb[];
  selected: string | null;
  max: number;
  onSelect: (id: string) => void;
  onAdd: (files: File[]) => void;
};

/** The outfits a customer wants to try, plus a tile that takes one or many photos at once. */
export function OutfitStrip({ outfits, selected, max, onSelect, onAdd }: Props) {
  const [over, setOver] = useState(false);
  const room = max - outfits.length;

  const take = (files: FileList | null) => {
    const images = Array.from(files ?? []).filter((f) => f.type.startsWith('image/'));
    if (images.length) onAdd(images);
  };

  return (
    <div className="outfits" role="listbox" aria-label="Outfits">
      {outfits.map((o, i) => (
        <button
          key={o.id}
          type="button"
          role="option"
          aria-selected={selected === o.id}
          className={`outfit ${selected === o.id ? 'on' : ''}`}
          onClick={() => onSelect(o.id)}
        >
          <img src={o.url} alt="" />
          <span className="outfit-no">{i + 1}</span>
          <span className="visually-hidden">{o.label}</span>
        </button>
      ))}
      {room > 0 && (
        <label
          htmlFor="outfit-add"
          className={`outfit add ${over ? 'over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            take(e.dataTransfer.files);
          }}
        >
          <span className="slot-plus" aria-hidden="true">+</span>
          <span className="slot-hint">{outfits.length ? 'Add more' : 'Add outfits'}</span>
          <input
            id="outfit-add"
            type="file"
            multiple
            accept="image/*"
            className="visually-hidden"
            onChange={(e) => {
              take(e.target.files);
              e.target.value = '';
            }}
          />
        </label>
      )}
    </div>
  );
}
