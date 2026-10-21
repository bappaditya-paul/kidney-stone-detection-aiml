import React from 'react';
import { Stethoscope, Info } from 'lucide-react';

interface ClinicalAdvisoryProps {
  advisoryText?: string;
  severityLevel?: string;
}

export const ClinicalAdvisory: React.FC<ClinicalAdvisoryProps> = ({
  advisoryText,
  severityLevel
}) => {
  return (
    <div className="card" style={{ marginTop: '1.5rem' }}>
      <div className="card-title">
        <Stethoscope size={22} color="var(--accent-teal)" />
        <span>Clinical Advisory & Recommendations</span>
      </div>

      <div className="advisory-box">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
          <Info size={18} />
          <span>Severity Assessment: {severityLevel || 'Evaluation in progress'}</span>
        </div>
        <p className="advisory-text">
          {advisoryText || 'Upload a scan and execute inference to generate structured medical guidance.'}
        </p>
      </div>

      <div style={{ marginTop: '1rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        ⚠️ <strong>Investigational Notice:</strong> NephroScan AI is designed as a computer-aided triage and clinical decision support system. Findings must be correlated with clinical history, urinalysis, and validated by a board-certified radiologist or urologist.
      </div>
    </div>
  );
};
