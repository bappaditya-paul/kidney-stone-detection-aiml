import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { Activity, ShieldCheck, Zap, Sparkles, RefreshCw, CheckCircle2, AlertOctagon } from 'lucide-react';
import { ScanUploader } from '@/components/ScanUploader';
import { HeatmapViewer } from '@/components/HeatmapViewer';
import { MetricCards } from '@/components/MetricCards';
import { ClinicalAdvisory } from '@/components/ClinicalAdvisory';

interface DiagnosticResult {
  is_valid_scan: boolean;
  label: string;
  probability: number;
  confidence: number;
  dilation_metrics?: {
    severity_level: string;
    dilation_status: string;
    hydronephrosis_risk: string;
    focal_stone_ratio: number;
    clinical_advisory: string;
  };
  gradcam_data_uri?: string;
  validation_message?: string;
  error?: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [validateModality, setValidateModality] = useState(true);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/health`)
      .then((res) => res.json())
      .then((data) => setApiOnline(data.status === 'healthy'))
      .catch(() => setApiOnline(false));
  }, [API_BASE]);

  const handleImageSelected = (file: File, url: string) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const runDiagnostics = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('validate_scan', String(validateModality));

    try {
      const response = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data: DiagnosticResult = await response.json();
      setResult(data);
    } catch (err: any) {
      alert(`Inference failed: ${err.message}. Ensure backend is running at ${API_BASE}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>NephroScan AI | Clinical Ultrasound Diagnostic Suite</title>
        <meta name="description" content="AI-assisted nephrolithiasis detection and pelvicalyceal dilation assessment" />
      </Head>

      {/* Navigation */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={24} />
          </div>
          <div>
            <span>NephroScan</span>
            <span style={{ color: 'var(--accent-cyan)' }}>.AI</span>
          </div>
          <span className="brand-badge">Clinical v1.0</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: apiOnline ? 'var(--success)' : 'var(--danger)',
                display: 'inline-block',
              }}
            />
            <span style={{ color: 'var(--text-secondary)' }}>
              API: {apiOnline === null ? 'Connecting...' : apiOnline ? 'Active (FastAPI)' : 'Offline (localhost:8000)'}
            </span>
          </div>
        </div>
      </nav>

      {/* Main Workspace */}
      <main className="container">
        {/* Hero Section */}
        <div className="hero-banner">
          <h1 className="hero-title">Ultrasound Nephrolithiasis Detection & XAI Cockpit</h1>
          <p className="hero-subtitle">
            Transfer Learning CNN architecture (MobileNetV2, 98.87% benchmark accuracy) with real-time Grad-CAM calculus localization and Out-of-Distribution modality guard.
          </p>
        </div>

        {/* Dashboard Grid */}
        <div className="dashboard-grid">
          {/* Left Column: Ingestion & Controls */}
          <div>
            <ScanUploader onImageSelected={handleImageSelected} isLoading={isLoading} />

            {/* Diagnostic Action Bar */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={validateModality}
                    onChange={(e) => setValidateModality(e.target.checked)}
                    style={{ accentColor: 'var(--accent-teal)' }}
                  />
                  <span>Enable Ultrasound Modality Guard (OOD Rejection)</span>
                </label>
              </div>

              <button
                className="btn-primary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={runDiagnostics}
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="animate-spin" size={18} />
                    <span>Analyzing Sonogram & Computing Grad-CAM...</span>
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    <span>Run Clinical Diagnostic Inference</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Column: Visualization & Heatmap */}
          <div>
            <HeatmapViewer
              originalImage={previewUrl}
              gradcamImage={result?.gradcam_data_uri || null}
              label={result?.label || null}
            />
          </div>
        </div>

        {/* Diagnostic Results Section */}
        {result && (
          <div style={{ marginTop: '2.5rem' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '1.25rem' }}>
              Diagnostic Evaluation & Quantitative Findings
            </h2>

            <MetricCards
              label={result.label}
              confidence={result.confidence}
              probability={result.probability}
              dilationStatus={result.dilation_metrics?.dilation_status}
              hydronephrosisRisk={result.dilation_metrics?.hydronephrosis_risk}
              isValidScan={result.is_valid_scan}
              errorMessage={result.error || result.validation_message}
            />

            {result.is_valid_scan && result.dilation_metrics && (
              <ClinicalAdvisory
                advisoryText={result.dilation_metrics.clinical_advisory}
                severityLevel={result.dilation_metrics.severity_level}
              />
            )}
          </div>
        )}
      </main>
    </>
  );
}
