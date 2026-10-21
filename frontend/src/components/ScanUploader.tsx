import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, ShieldCheck, AlertCircle } from 'lucide-react';

interface ScanUploaderProps {
  onImageSelected: (file: File, previewUrl: string) => void;
  isLoading: boolean;
}

export const ScanUploader: React.FC<ScanUploaderProps> = ({ onImageSelected, isLoading }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPEG, PNG).');
      return;
    }
    setSelectedFileName(file.name);
    const previewUrl = URL.createObjectURL(file);
    onImageSelected(file, previewUrl);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <UploadCloud size={22} color="var(--accent-cyan)" />
        <span>Ultrasound Image Ingestion</span>
      </div>

      <div
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="dropzone-icon">
          <FileImage size={28} />
        </div>
        <p style={{ fontWeight: 600, marginBottom: '0.35rem' }}>
          {selectedFileName ? selectedFileName : 'Drop B-mode kidney ultrasound scan here'}
        </p>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Supports DICOM-exported JPEG, PNG • Analyzed with OOD Guard
        </p>
      </div>

      <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
        <ShieldCheck size={16} color="var(--accent-teal)" />
        <span>Ultrasound Modality Guard enabled: Blocks non-medical photos automatically</span>
      </div>
    </div>
  );
};
