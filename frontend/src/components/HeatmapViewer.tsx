import React from 'react';
import { Eye, Layers } from 'lucide-react';

interface HeatmapViewerProps {
  originalImage: string | null;
  gradcamImage: string | null;
  label: string | null;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalImage,
  gradcamImage,
  label
}) => {
  if (!originalImage) {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '360px', color: 'var(--text-muted)' }}>
        <Layers size={48} strokeWidth={1.5} style={{ marginBottom: '1rem', opacity: 0.4 }} />
        <p style={{ fontWeight: 500 }}>No scan uploaded</p>
        <p style={{ fontSize: '0.82rem' }}>Upload a kidney ultrasound image to view Grad-CAM explainability</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        <Eye size={22} color="var(--accent-cyan)" />
        <span>Diagnostic Imaging & Activation Heatmap</span>
      </div>

      <div className="viewer-container">
        {/* Original Scan */}
        <div className="viewer-frame">
          <div className="viewer-tag">Original B-Mode Scan</div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={originalImage} alt="Original Ultrasound" />
        </div>

        {/* Grad-CAM Heatmap */}
        <div className="viewer-frame">
          <div className="viewer-tag">Grad-CAM (out_relu)</div>
          {gradcamImage ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={gradcamImage} alt="Grad-CAM Activation Map" />
          ) : (
            <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Run inference to generate calculus focus heatmap
            </div>
          )}
        </div>
      </div>

      {gradcamImage && (
        <div style={{ marginTop: '0.9rem', fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
          <span>Jet Colormap: Red = Highest focal attention</span>
          <span>Layer: MobileNetV2 `out_relu`</span>
        </div>
      )}
    </div>
  );
};
