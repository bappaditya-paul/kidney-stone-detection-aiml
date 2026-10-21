import React from 'react';
import { Activity, AlertTriangle, CheckCircle, Percent, ShieldAlert } from 'lucide-react';

interface MetricCardsProps {
  label: string;
  confidence: number;
  probability: number;
  dilationStatus?: string;
  hydronephrosisRisk?: string;
  isValidScan: boolean;
  errorMessage?: string | null;
}

export const MetricCards: React.FC<MetricCardsProps> = ({
  label,
  confidence,
  probability,
  dilationStatus,
  hydronephrosisRisk,
  isValidScan,
  errorMessage
}) => {
  if (!isValidScan) {
    return (
      <div className="verdict-banner verdict-warning">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <ShieldAlert size={28} />
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.05rem' }}>Non-Medical Scan Rejected (OOD Guard)</div>
            <div style={{ fontSize: '0.84rem' }}>{errorMessage || 'Image failed ultrasound modality criteria.'}</div>
          </div>
        </div>
      </div>
    );
  }

  const isStone = label === 'Stone';

  return (
    <div>
      {/* Primary Diagnostic Banner */}
      <div className={`verdict-banner ${isStone ? 'verdict-positive' : 'verdict-negative'}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          {isStone ? <AlertTriangle size={32} /> : <CheckCircle size={32} />}
          <div>
            <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.9 }}>
              AI Diagnostic Verdict
            </div>
            <div style={{ fontSize: '1.65rem', fontWeight: 800 }}>
              {isStone ? 'Nephrolithiasis Detected (Calculus Present)' : 'No Calculus Detected (Unremarkable)'}
            </div>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.8rem', opacity: 0.9 }}>Diagnostic Confidence</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800 }}>{confidence.toFixed(1)}%</div>
        </div>
      </div>

      {/* Numerical Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-stat">
          <div className="metric-label">Model Probability</div>
          <div className="metric-value" style={{ color: isStone ? 'var(--danger)' : 'var(--success)' }}>
            {(probability * 100).toFixed(2)}%
          </div>
        </div>

        <div className="metric-stat">
          <div className="metric-label">Hydronephrosis Risk</div>
          <div className="metric-value" style={{ color: hydronephrosisRisk?.includes('Elevated') ? 'var(--danger)' : 'var(--accent-cyan)' }}>
            {hydronephrosisRisk || 'Low (0%)'}
          </div>
        </div>

        <div className="metric-stat" style={{ gridColumn: 'span 2' }}>
          <div className="metric-label">Pelvicalyceal Architecture & Dilation</div>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#e2e8f0', marginTop: '0.2rem' }}>
            {dilationStatus || 'Normal Architecture (No Dilation)'}
          </div>
        </div>
      </div>
    </div>
  );
};
