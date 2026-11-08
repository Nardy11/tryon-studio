import { useState } from 'react';

type Props = {
  id: string;
  label: string;
  hint: string;
  url?: string;
  contain?: boolean;
  onFile: (file: File | null) => void;
};

/** A drop zone + file picker that shows the chosen photo. */
export function PhotoSlot({ id, label, hint, url, contain, onFile }: Props) {
  const [over, setOver] = useState(false);

  const take = (files: FileList | null) => {
    const file = files?.[0];
    if (file && file.type.startsWith('image/')) onFile(file);
  };

  return (
    <div className={`slot ${url ? 'filled' : ''} ${over ? 'over' : ''}`}>
      <label
        htmlFor={id}
        className="slot-drop"
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
        {url ? (
          <img src={url} alt={`${label} photo`} className={contain ? 'contain' : ''} />
        ) : (
          <span className="slot-empty">
            <span className="slot-plus" aria-hidden="true">+</span>
            <span className="slot-hint">{hint}</span>
          </span>
        )}
      </label>
      <input
        id={id}
        type="file"
        accept="image/*"
        className="visually-hidden"
        onChange={(e) => {
          take(e.target.files);
          e.target.value = '';
        }}
      />
      <div className="slot-foot">
        <span className="slot-label">{label}</span>
        {url && (
          <button type="button" className="link" onClick={() => onFile(null)}>
            Remove
          </button>
        )}
      </div>
    </div>
  );
}
