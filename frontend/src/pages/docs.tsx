import React, { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { 
  FileText, ShieldAlert, Cpu, Layers, Activity, 
  Code, ShieldCheck, Database, Github, ArrowLeft, 
  AlertTriangle, CheckCircle2, ChevronRight, Layout,
  ExternalLink, Copy, Check, Terminal, Sparkles
} from "lucide-react";

export default function DocsPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const sidebarLinks = [
    { id: "overview", icon: FileText, label: "Project Overview" },
    { id: "modality-guard", icon: ShieldAlert, label: "Modality Guard (OOD)" },
    { id: "dataset", icon: Database, label: "Training Data & Hygiene" },
    { id: "architecture", icon: Cpu, label: "Model Architecture" },
    { id: "grad-cam", icon: Layers, label: "Grad-CAM Explainability" },
    { id: "dilation", icon: Activity, label: "Pelvicalyceal Grading" },
    { id: "fastapi", icon: Code, label: "FastAPI Microservice" },
    { id: "frontend-guide", icon: Layout, label: "Opensource UI Guide" },
  ];

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <>
      <Head>
        <title>Technical & Clinical Documentation | KidneyAI</title>
        <meta 
          name="description" 
          content="Comprehensive first-principles engineering and clinical documentation for KidneyAI: Modality Guard, MobileNetV2 architecture, Grad-CAM XAI, and FastAPI microservice." 
        />
      </Head>

      <div className="min-h-screen bg-white text-zinc-900 font-sans flex flex-col selection:bg-zinc-200">
        
        {/* ==================== TOP NAVIGATION ==================== */}
        <header className="w-full border-b border-neutral-200 bg-white/95 backdrop-blur-md sticky top-0 z-30 h-16 flex items-center justify-between px-4 md:px-8">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5 group">
              <ShieldCheck className="text-emerald-600 transition-transform group-hover:scale-105" size={24} />
              <span className="font-serif text-2xl tracking-tight mt-0.5 text-zinc-900">
                KidneyAI
              </span>
            </Link>
            <span className="hidden sm:inline-block ml-2 px-2 py-0.5 rounded-md bg-zinc-100 border border-neutral-200 text-xs font-semibold text-zinc-600 tracking-wide uppercase">
              Docs Portal
            </span>
          </div>
          
          <nav className="flex items-center gap-5 text-sm font-medium text-zinc-600">
            <Link 
              href="/" 
              className="flex items-center gap-1.5 text-zinc-900 hover:text-emerald-600 transition-colors"
            >
              <ArrowLeft size={16} /> <span className="hidden sm:inline">Diagnostic Scanner</span>
            </Link>
            <a 
              href="https://www.kaggle.com/datasets/jayaprakashpondy/kidney-cancer-image" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors"
            >
              <Database size={16} /> <span className="hidden sm:inline">Kaggle Dataset</span>
              <ExternalLink size={12} className="text-zinc-400" />
            </a>
            <a 
              href="https://github.com/bappaditya-paul/kidney-stone-detection-aiml" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors"
            >
              <Github size={16} /> <span className="hidden sm:inline">GitHub</span>
            </a>
          </nav>
        </header>

        {/* ==================== WORKSPACE LAYOUT ==================== */}
        <div className="flex flex-1 overflow-hidden">
          
          {/* ==================== LEFT SIDEBAR ==================== */}
          <aside className="w-72 border-r border-neutral-200 bg-zinc-50/70 flex-shrink-0 hidden md:block overflow-y-auto">
            <div className="p-5">
              <div className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider mb-3 px-3">
                Core Documentation
              </div>
              <nav className="space-y-1">
                {sidebarLinks.map((link) => {
                  const isActive = activeTab === link.id;
                  const Icon = link.icon;
                  return (
                    <button
                      key={link.id}
                      onClick={() => setActiveTab(link.id)}
                      className={`w-full flex items-center gap-3 px-3.5 py-2.5 text-sm rounded-xl transition-all text-left ${
                        isActive 
                          ? "bg-white border border-neutral-200/90 text-zinc-900 font-semibold shadow-xs" 
                          : "border border-transparent text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
                      }`}
                    >
                      <Icon size={17} className={isActive ? "text-emerald-600" : "text-zinc-400"} />
                      <span>{link.label}</span>
                    </button>
                  );
                })}
              </nav>

              <div className="mt-8 pt-6 border-t border-neutral-200/80 px-3">
                <div className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider mb-2">
                  System Status
                </div>
                <div className="p-3 bg-white border border-neutral-200 rounded-xl space-y-1.5 text-xs">
                  <div className="flex items-center justify-between text-zinc-600">
                    <span>Model Engine:</span>
                    <span className="font-medium text-emerald-600">MobileNetV2</span>
                  </div>
                  <div className="flex items-center justify-between text-zinc-600">
                    <span>Holdout Accuracy:</span>
                    <span className="font-mono font-semibold text-zinc-900">98.87%</span>
                  </div>
                  <div className="flex items-center justify-between text-zinc-600">
                    <span>Modality Guard:</span>
                    <span className="font-semibold text-emerald-600">Active (4-Tier)</span>
                  </div>
                </div>
              </div>
            </div>
          </aside>

          {/* ==================== MAIN CONTENT VIEWPORT ==================== */}
          <main className="flex-1 overflow-y-auto bg-white">
            <div className="max-w-4xl mx-auto py-10 px-6 md:py-14 md:px-12">
              
              {/* ==================== TAB 1: PROJECT OVERVIEW ==================== */}
              {activeTab === "overview" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-emerald-800 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Clinical AI Specification
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      NephroScan AI / KidneyAI Overview
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      KidneyAI is an open-source deep learning clinical decision support system engineered for high-accuracy renal calculus (kidney stone) detection and secondary pelvicalyceal dilation assessment from B-mode ultrasound scans.
                    </p>
                  </div>

                  {/* Summary Metric Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="p-5 border border-neutral-200 rounded-2xl bg-zinc-50/50">
                      <div className="text-xs text-zinc-500 font-medium uppercase tracking-wider">Independent Test Accuracy</div>
                      <div className="text-3xl font-serif font-bold text-zinc-900 mt-1">98.87%</div>
                      <div className="text-xs text-emerald-600 font-medium mt-1">Evaluated on 1,676 holdout scans</div>
                    </div>
                    <div className="p-5 border border-neutral-200 rounded-2xl bg-zinc-50/50">
                      <div className="text-xs text-zinc-500 font-medium uppercase tracking-wider">Weighted F1-Score</div>
                      <div className="text-3xl font-serif font-bold text-zinc-900 mt-1">0.99</div>
                      <div className="text-xs text-zinc-600 font-medium mt-1">Harmonic mean of precision & recall</div>
                    </div>
                    <div className="p-5 border border-neutral-200 rounded-2xl bg-zinc-50/50">
                      <div className="text-xs text-zinc-500 font-medium uppercase tracking-wider">Ultrasound Modality Guard</div>
                      <div className="text-3xl font-serif font-bold text-emerald-600 mt-1">4-Stage</div>
                      <div className="text-xs text-zinc-600 font-medium mt-1">OOD physical invariant filtering</div>
                    </div>
                  </div>

                  {/* Clinical Problem vs Solution */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      1. The Clinical Problem & Ultrasound Physics
                    </h2>
                    <p className="text-sm md:text-base text-zinc-600 leading-relaxed">
                      <strong>Nephrolithiasis</strong> affects over 10% of the population, causing intense acute renal colic when mineral concretions obstruct the urinary collecting system. While non-contrast CT scans (NCCT) are historically sensitive, clinical guidelines from the <strong>European Association of Urology (EAU)</strong> recommend ultrasound as the primary initial imaging modality due to <strong>zero ionizing radiation</strong> and bedside point-of-care (POCUS) availability.
                    </p>

                    <div className="p-5 border border-neutral-200 rounded-2xl bg-white shadow-xs space-y-3">
                      <h3 className="text-sm font-bold text-zinc-900 uppercase tracking-wide">
                        The Dual Sonographic Hallmark
                      </h3>
                      <p className="text-sm text-zinc-600 leading-relaxed">
                        In B-mode ultrasound, renal stones exhibit two distinct, correlated physical phenomena:
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                        <div className="p-4 bg-zinc-50 border border-neutral-200 rounded-xl">
                          <div className="text-xs font-bold text-zinc-900 mb-1">1. Hyperechoic Reflection</div>
                          <div className="text-xs text-zinc-600 leading-relaxed">
                            Acoustic impedance mismatch between soft parenchyma and dense mineral crystalline matrices generates a localized, bright white echo peak.
                          </div>
                        </div>
                        <div className="p-4 bg-zinc-50 border border-neutral-200 rounded-xl">
                          <div className="text-xs font-bold text-zinc-900 mb-1">2. Posterior Acoustic Shadow</div>
                          <div className="text-xs text-zinc-600 leading-relaxed">
                            Near-total sound beam attenuation and reflection creates an anechoic (jet-black) shadow band directly distal to the stone focus.
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-end">
                    <button
                      onClick={() => setActiveTab("modality-guard")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Modality Guard (OOD)</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 2: MODALITY GUARD (OOD) ==================== */}
              {activeTab === "modality-guard" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-rose-800 bg-rose-50 border border-rose-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Physics-Based Invariant Filter
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Ultrasound Modality Guard
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      How KidneyAI prevents out-of-distribution (OOD) false positives by verifying acoustic physical invariants before running deep neural network inference.
                    </p>
                  </div>

                  {/* Warning Callout Box */}
                  <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
                    <h4 className="flex items-center gap-2 text-amber-900 font-semibold mb-2 text-sm uppercase tracking-wide">
                      <AlertTriangle size={18} />
                      The Naive Approach Failure: The &quot;Panda Photo&quot; Anomaly
                    </h4>
                    <p className="text-sm text-amber-800 leading-relaxed mb-3">
                      Standard binary CNNs use a sigmoid activation: <code className="font-mono font-semibold">σ(W · x + b) ∈ [0.0, 1.0]</code>. They operate under a <strong>closed-world assumption</strong>: every input image is assumed to be a renal scan.
                    </p>
                    <p className="text-sm text-amber-800 leading-relaxed">
                      When users upload photos of giant pandas, cars, or documents, convolutional edge filters respond to high-contrast black-and-white borders (e.g. dark panda eyes against white fur), erroneously outputting <strong>100% Calculus Confidence</strong>!
                    </p>
                  </div>

                  {/* The 4-Stage Invariant Pipeline */}
                  <div className="space-y-6">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      The 4-Stage Acoustic Physics Filter
                    </h2>
                    
                    <div className="space-y-4">
                      {/* Gate 1 */}
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-zinc-900 uppercase">Stage 1: HSV Color Saturation Gate</span>
                          <span className="font-mono text-xs text-zinc-500">Mean(S) ≤ 0.18</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Authentic B-mode ultrasound is strictly monochromatic. In the HSV color space, the saturation channel <code className="font-mono">S = 1 - min(R,G,B)/max(R,G,B)</code> measures color purity. Scans with chromatic color shifts or vivid consumer photography are instantly rejected.
                        </p>
                      </div>

                      {/* Gate 2 */}
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-zinc-900 uppercase">Stage 2: RGB Inter-Channel Divergence Gate</span>
                          <span className="font-mono text-xs text-zinc-500">Δ_RGB ≤ 16.0</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          In genuine ultrasound scans, the Red, Green, and Blue phosphor values are identical: <code className="font-mono">R ≈ G ≈ B</code>. We compute <code className="font-mono">mean(|R-G| + |G-B| + |B-R|)</code>. Any value exceeding 16.0 indicates non-medical lighting or colored artwork.
                        </p>
                      </div>

                      {/* Gate 3 */}
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-zinc-900 uppercase">Stage 3: Acoustic Speckle Texture Gate (Laplacian Variance)</span>
                          <span className="font-mono text-xs text-zinc-500">Var(∇²I) ≥ 8.0</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          The microscopic cellular heterogeneity of renal parenchyma causes acoustic wave interference, producing signature <strong>Rayleigh speckle noise</strong>. By convolving with a discrete 3×3 Laplacian operator <code className="font-mono">∇²I</code>, we measure texture variance. Flat digital graphics, cartoons, and documents produce near-zero variance and are rejected.
                        </p>
                      </div>

                      {/* Gate 4 */}
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-zinc-900 uppercase">Stage 4: Dynamic Intensity & Contrast Gate</span>
                          <span className="font-mono text-xs text-zinc-500">StdDev ≥ 15.0 & 8.0 ≤ Mean ≤ 235.0</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Eliminates blank probe-off frames, completely black DICOM files, or overexposed scanner screen captures.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("overview")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Overview</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("dataset")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Training Data & Hygiene</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 3: DATASET & HYGIENE ==================== */}
              {activeTab === "dataset" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-zinc-600 bg-zinc-100 border border-neutral-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Dataset & Methodology
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Training Data & Clinical Hygiene
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      The KidneyAI model was trained, validated, and evaluated on a benchmark renal ultrasound dataset comprising 26,168 clinical images from Kaggle, subjected to strict deduplication and patient-leakage prevention.
                    </p>
                  </div>

                  {/* Kaggle Link Pill */}
                  <div className="flex flex-wrap items-center gap-4">
                    <a 
                      href="https://www.kaggle.com/datasets/jayaprakashpondy/kidney-cancer-image" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <Database size={16} />
                      Open Dataset on Kaggle
                      <ExternalLink size={14} className="text-zinc-400" />
                    </a>
                    <span className="text-xs text-zinc-500 font-medium flex items-center gap-1.5">
                      <CheckCircle2 size={16} className="text-emerald-500" />
                      26,168 Ultrasound Scans
                    </span>
                  </div>

                  {/* Clinical Context Box */}
                  <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
                    <h4 className="flex items-center gap-2 text-amber-900 font-semibold mb-2 text-sm uppercase tracking-wide">
                      <AlertTriangle size={18} />
                      Critical Context: Preventing Data Leakage
                    </h4>
                    <p className="text-sm text-amber-800 leading-relaxed mb-3">
                      In medical AI, data leakage (where train and test sets contain duplicate or near-identical images from the same patient) artificially inflates accuracy and leads to catastrophic clinical failure.
                    </p>
                    <ul className="text-xs text-amber-900 list-disc list-inside space-y-1.5">
                      <li><strong>Automated SHA-256 Deduplication:</strong> Removed all exact duplicate frames using hash comparison.</li>
                      <li><strong>Corrupted Frame Purging:</strong> Removed truncated or unexposed probe-off frames.</li>
                      <li><strong>Strict Holdout Partition:</strong> An independent test set of 1,676 scans was completely frozen prior to model hyperparameter tuning.</li>
                    </ul>
                  </div>

                  {/* Dataset Partition Table */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      Dataset Composition & Split Statistics
                    </h2>
                    
                    <div className="border border-neutral-200 rounded-2xl overflow-hidden bg-white shadow-xs">
                      <table className="w-full text-left text-sm">
                        <thead className="bg-zinc-50 border-b border-neutral-200 text-xs font-semibold text-zinc-600">
                          <tr>
                            <th className="p-4">Partition Set</th>
                            <th className="p-4">Normal (Class 0)</th>
                            <th className="p-4">Calculus (Class 1)</th>
                            <th className="p-4">Total Scans</th>
                            <th className="p-4">Ratio</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-100 font-mono text-xs">
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Training Split</td>
                            <td className="p-4 text-zinc-700">5,364</td>
                            <td className="p-4 text-zinc-700">5,364</td>
                            <td className="p-4 font-semibold text-zinc-900">10,728</td>
                            <td className="p-4 text-zinc-500">64.0%</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Validation Split</td>
                            <td className="p-4 text-zinc-700">1,341</td>
                            <td className="p-4 text-zinc-700">1,341</td>
                            <td className="p-4 font-semibold text-zinc-900">2,682</td>
                            <td className="p-4 text-zinc-500">16.0%</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Holdout Test Set</td>
                            <td className="p-4 text-zinc-700">838</td>
                            <td className="p-4 text-zinc-700">838</td>
                            <td className="p-4 font-semibold text-zinc-900">1,676</td>
                            <td className="p-4 text-zinc-500">20.0%</td>
                          </tr>
                          <tr className="bg-zinc-50/50 font-bold font-sans">
                            <td className="p-4 text-zinc-900">Total Curated Dataset</td>
                            <td className="p-4 font-mono text-zinc-900">7,543</td>
                            <td className="p-4 font-mono text-zinc-900">7,543</td>
                            <td className="p-4 font-mono text-zinc-900">15,086</td>
                            <td className="p-4 font-mono text-zinc-900">100%</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("modality-guard")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Modality Guard</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("architecture")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Model Architecture</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 4: MODEL ARCHITECTURE ==================== */}
              {activeTab === "architecture" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-blue-800 bg-blue-50 border border-blue-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Deep Learning Engineering
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Model Architecture & Transfer Learning
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      Detailed layer topology, inverted residual bottlenecks, depthwise separable convolutions, and computational FLOP reduction.
                    </p>
                  </div>

                  {/* Why MobileNetV2 */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      Why MobileNetV2 Over Heavy CNNs?
                    </h2>
                    <p className="text-sm md:text-base text-zinc-600 leading-relaxed">
                      Traditional architectures like ResNet-152 or VGG-19 possess tens of millions of parameters. In medical ultrasound, where cohort sizes are constrained, large parameter counts cause networks to memorize scanner vendor watermarks rather than pathology.
                    </p>
                    <p className="text-sm md:text-base text-zinc-600 leading-relaxed">
                      MobileNetV2 splits standard convolutions into <strong>depthwise separable convolutions</strong>:
                    </p>

                    <div className="p-5 border border-neutral-200 rounded-2xl bg-zinc-50/60 font-mono text-xs space-y-2">
                      <div className="text-zinc-500 font-sans font-medium uppercase text-[11px]">FLOP Reduction Proof:</div>
                      <div>Standard Conv Cost = h · w · Dk² · M · N</div>
                      <div>Separable Conv Cost = h · w · Dk² · M + h · w · M · N</div>
                      <div className="text-emerald-700 font-bold">
                        Cost Ratio ≈ 1/N + 1/Dk² ≈ 1/3² = 1/9 (88% computation reduction!)
                      </div>
                    </div>
                  </div>

                  {/* Layer Topology Table */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      Layer-by-Layer Tensor Pipeline
                    </h2>
                    
                    <div className="border border-neutral-200 rounded-2xl overflow-hidden bg-white shadow-xs">
                      <table className="w-full text-left text-sm">
                        <thead className="bg-zinc-50 border-b border-neutral-200 text-xs font-semibold text-zinc-600">
                          <tr>
                            <th className="p-4">Stage / Layer</th>
                            <th className="p-4">Output Dimensions</th>
                            <th className="p-4">Parameters</th>
                            <th className="p-4">Function</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-100 font-mono text-xs">
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Input Layer</td>
                            <td className="p-4 text-zinc-700">(None, 224, 224, 3)</td>
                            <td className="p-4 text-zinc-500">0</td>
                            <td className="p-4 font-sans text-zinc-600">Normalized RGB scan</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">MobileNetV2 Base</td>
                            <td className="p-4 text-zinc-700">(None, 7, 7, 1280)</td>
                            <td className="p-4 text-zinc-500">2,257,984</td>
                            <td className="p-4 font-sans text-zinc-600">Feature extractor (ImageNet weights)</td>
                          </tr>
                          <tr className="bg-emerald-50/40">
                            <td className="p-4 font-sans font-bold text-emerald-900">Target Layer: out_relu</td>
                            <td className="p-4 font-bold text-emerald-900">(None, 7, 7, 1280)</td>
                            <td className="p-4 text-zinc-500">—</td>
                            <td className="p-4 font-sans text-emerald-800">Target for Grad-CAM gradient tape</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">GlobalAveragePooling2D</td>
                            <td className="p-4 text-zinc-700">(None, 1280)</td>
                            <td className="p-4 text-zinc-500">0</td>
                            <td className="p-4 font-sans text-zinc-600">Spatial feature averaging</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Dropout (0.3)</td>
                            <td className="p-4 text-zinc-700">(None, 1280)</td>
                            <td className="p-4 text-zinc-500">0</td>
                            <td className="p-4 font-sans text-zinc-600">Regularization against overfitting</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Dense (128, ReLU)</td>
                            <td className="p-4 text-zinc-700">(None, 128)</td>
                            <td className="p-4 text-zinc-500">163,968</td>
                            <td className="p-4 font-sans text-zinc-600">Nonlinear feature mapping</td>
                          </tr>
                          <tr>
                            <td className="p-4 font-sans font-medium text-zinc-900">Dense (1, Sigmoid)</td>
                            <td className="p-4 text-zinc-700">(None, 1)</td>
                            <td className="p-4 text-zinc-500">129</td>
                            <td className="p-4 font-sans text-zinc-600">Calculus probability score [0.0 - 1.0]</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("dataset")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Training Data</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("grad-cam")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Grad-CAM Explainability</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 5: GRAD-CAM EXPLAINABILITY ==================== */}
              {activeTab === "grad-cam" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-purple-800 bg-purple-50 border border-purple-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Explainable AI (XAI)
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Grad-CAM Explainability
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      How KidneyAI generates class-discriminative spatial heatmaps to prove the AI is focusing on the hyperechoic calculus and acoustic shadow rather than background noise.
                    </p>
                  </div>

                  {/* Clinical Value Callout */}
                  <div className="bg-zinc-50 border border-neutral-200 rounded-2xl p-6 space-y-2">
                    <h4 className="font-bold text-zinc-900 text-sm uppercase tracking-wide">
                      Clinical Necessity of Explainable AI
                    </h4>
                    <p className="text-sm text-zinc-600 leading-relaxed">
                      A doctor cannot trust a raw probability score (e.g. <code className="font-mono">Stone: 96%</code>). In clinical sonography, knowing <em>where</em> the model detected the abnormality is essential for distinguishing actual renal calculi from arterial wall calcifications, pelvic cysts, or bowel gas artifacts.
                    </p>
                  </div>

                  {/* Mathematical Derivation */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      Step-by-Step Mathematical Flow
                    </h2>

                    <div className="space-y-3 text-sm text-zinc-600 leading-relaxed">
                      <div className="p-4 border border-neutral-200 rounded-xl bg-white space-y-1">
                        <div className="text-xs font-bold text-zinc-900 uppercase">Step 1: Backward Gradient Computation</div>
                        <p className="text-xs text-zinc-600">
                          Let <code className="font-mono">y^c</code> be the raw calculus prediction score. We compute the gradient of <code className="font-mono">y^c</code> with respect to the feature map activations <code className="font-mono">A^k</code> of layer <code className="font-mono">out_relu</code> (shape 7×7×1280).
                        </p>
                      </div>

                      <div className="p-4 border border-neutral-200 rounded-xl bg-white space-y-1">
                        <div className="text-xs font-bold text-zinc-900 uppercase">Step 2: Global Average Pooling of Gradients</div>
                        <p className="text-xs text-zinc-600">
                          Neuron importance weights <code className="font-mono">α_k^c</code> are obtained by averaging gradients over spatial dimensions (Z = 7 × 7 = 49):
                        </p>
                        <div className="p-2 bg-zinc-50 font-mono text-xs rounded border border-neutral-200">
                          α_k^c = (1 / Z) · ∑_i ∑_j ( ∂y^c / ∂A_ij^k )
                        </div>
                      </div>

                      <div className="p-4 border border-neutral-200 rounded-xl bg-white space-y-1">
                        <div className="text-xs font-bold text-zinc-900 uppercase">Step 3: Weighted Linear Combination & ReLU Activation</div>
                        <p className="text-xs text-zinc-600">
                          We compute a weighted sum of feature maps and pass it through ReLU to isolate features that <strong>increase</strong> the likelihood of stone presence:
                        </p>
                        <div className="p-2 bg-zinc-50 font-mono text-xs rounded border border-neutral-200">
                          L_GradCAM^c = ReLU( ∑_k α_k^c · A^k )
                        </div>
                      </div>

                      <div className="p-4 border border-neutral-200 rounded-xl bg-white space-y-1">
                        <div className="text-xs font-bold text-zinc-900 uppercase">Step 4: Alpha-Composited Jet Blending</div>
                        <p className="text-xs text-zinc-600">
                          The normalized localization map is upsampled to the original ultrasound dimensions and blended with alpha = 0.35:
                        </p>
                        <div className="p-2 bg-zinc-50 font-mono text-xs rounded border border-neutral-200">
                          Composite_Image = 0.65 · Original_Scan + 0.35 · Jet_Colormap(L_GradCAM)
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("architecture")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Architecture</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("dilation")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Pelvicalyceal Grading</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 6: PELVICALYCEAL DILATION ==================== */}
              {activeTab === "dilation" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Secondary Complication Engine
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Pelvicalyceal Dilation & Severity Index
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      Correlating calculus probability with spatial activation footprint to grade obstructive hydronephrosis risk and recommend clinical action protocols.
                    </p>
                  </div>

                  {/* 4-Tier Severity Cards */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      The 4 Clinical Severity Tiers
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-emerald-700 uppercase">Tier 1: Normal / Unremarkable</span>
                          <span className="font-mono text-xs text-zinc-500">P &lt; 0.50</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Baseline normal renal architecture. Central renal echo complex remains compact and uninterrupted. Low hydronephrosis risk (0%). Routine outpatient care.
                        </p>
                      </div>

                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-amber-700 uppercase">Tier 2: Borderline / Small Calculus</span>
                          <span className="font-mono text-xs text-zinc-500">0.50 ≤ P ≤ 0.65</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Minimal pelvic separation (&le; 5mm). Possible micro-calculus or anatomical variant. Moderate hydronephrosis risk (~25%). Recommend hydration and follow-up scan.
                        </p>
                      </div>

                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-orange-700 uppercase">Tier 3: Moderate Calculus</span>
                          <span className="font-mono text-xs text-zinc-500">0.65 &lt; P ≤ 0.85</span>
                        </div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Mild-to-moderate pelvicalyceal separation (~10-15mm). Definite hyperechoic focus. Moderate hydronephrosis risk (~50%). Confirmatory non-contrast CT KUB recommended.
                        </p>
                      </div>

                      <div className="p-5 border border-rose-200 rounded-2xl bg-rose-50/40 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-rose-700 uppercase">Tier 4: Acute Obstructive Calculus</span>
                          <span className="font-mono text-xs text-rose-600">P &gt; 0.85</span>
                        </div>
                        <p className="text-xs text-rose-900 leading-relaxed">
                          Marked caliceal ballooning with dense posterior acoustic shadowing. High hydronephrosis risk (&gt;75%). Urgent urology consultation for stenting or lithotripsy.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("grad-cam")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Grad-CAM</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("fastapi")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: FastAPI Reference</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 7: FASTAPI MICROSERVICE ==================== */}
              {activeTab === "fastapi" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-zinc-600 bg-zinc-100 border border-neutral-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Backend Architecture
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      FastAPI Microservice Reference
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      Asynchronous REST endpoints running on Port 8000 with singleton model caching, Pydantic v2 schemas, and zero redundant disk I/O.
                    </p>
                  </div>

                  {/* Endpoints */}
                  <div className="space-y-6">
                    {/* Predict Endpoint */}
                    <div className="border border-neutral-200 rounded-2xl overflow-hidden bg-white shadow-xs">
                      <div className="p-4 bg-zinc-50 border-b border-neutral-200 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="px-2.5 py-1 bg-emerald-100 text-emerald-800 rounded-lg text-xs font-mono font-bold">
                            POST
                          </span>
                          <span className="font-mono text-xs font-semibold text-zinc-900">
                            /api/v1/predict
                          </span>
                        </div>
                        <span className="text-xs text-zinc-500 font-medium">Full Diagnostic Inference</span>
                      </div>
                      
                      <div className="p-5 space-y-4 text-xs">
                        <p className="text-zinc-600">
                          Takes an uploaded ultrasound scan file, executes the 4-stage Modality Guard, runs MobileNetV2 inference, generates Grad-CAM heatmap Base64, and grades pelvicalyceal dilation.
                        </p>

                        <div>
                          <div className="flex items-center justify-between text-zinc-500 font-mono uppercase text-[11px] mb-1.5">
                            <span>Sample cURL Request</span>
                            <button
                              onClick={() => handleCopy(`curl -X POST "http://localhost:8000/api/v1/predict" -F "file=@kidney_scan.jpg" -F "validate_scan=true"`, "curl-predict")}
                              className="flex items-center gap-1 text-zinc-600 hover:text-zinc-900"
                            >
                              {copiedCode === "curl-predict" ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                              <span>{copiedCode === "curl-predict" ? "Copied" : "Copy"}</span>
                            </button>
                          </div>
                          <pre className="p-3 bg-zinc-900 text-zinc-100 rounded-xl overflow-x-auto font-mono">
{`curl -X POST "http://localhost:8000/api/v1/predict" \\
     -F "file=@kidney_scan.jpg;type=image/jpeg" \\
     -F "validate_scan=true"`}
                          </pre>
                        </div>
                      </div>
                    </div>

                    {/* Validate Endpoint */}
                    <div className="border border-neutral-200 rounded-2xl overflow-hidden bg-white shadow-xs">
                      <div className="p-4 bg-zinc-50 border-b border-neutral-200 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="px-2.5 py-1 bg-blue-100 text-blue-800 rounded-lg text-xs font-mono font-bold">
                            POST
                          </span>
                          <span className="font-mono text-xs font-semibold text-zinc-900">
                            /api/v1/validate
                          </span>
                        </div>
                        <span className="text-xs text-zinc-500 font-medium">Fast Pre-Screening</span>
                      </div>
                      <div className="p-5 text-xs text-zinc-600">
                        Evaluates physical acoustic invariants in under 5ms without loading or running the neural network.
                      </div>
                    </div>

                    {/* Health Endpoint */}
                    <div className="border border-neutral-200 rounded-2xl overflow-hidden bg-white shadow-xs">
                      <div className="p-4 bg-zinc-50 border-b border-neutral-200 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="px-2.5 py-1 bg-zinc-200 text-zinc-800 rounded-lg text-xs font-mono font-bold">
                            GET
                          </span>
                          <span className="font-mono text-xs font-semibold text-zinc-900">
                            /api/v1/health
                          </span>
                        </div>
                        <span className="text-xs text-zinc-500 font-medium">Service Readiness Probe</span>
                      </div>
                      <div className="p-5 text-xs text-zinc-600 font-mono">
                        {`{ "status": "healthy", "model_loaded": true, "version": "1.0.0" }`}
                      </div>
                    </div>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("dilation")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: Pelvicalyceal Grading</span>
                    </button>
                    <button
                      onClick={() => setActiveTab("frontend-guide")}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-xl hover:bg-zinc-800 transition-colors"
                    >
                      <span>Next: Opensource UI Guide</span>
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}

              {/* ==================== TAB 8: OPENSOURCE UI GUIDE ==================== */}
              {activeTab === "frontend-guide" && (
                <div className="space-y-10 animate-fade-in">
                  <div>
                    <span className="inline-flex items-center text-xs font-semibold tracking-wide text-zinc-700 bg-zinc-100 border border-neutral-200 px-2.5 py-1 rounded-md uppercase mb-4">
                      Frontend Architecture
                    </span>
                    <h1 className="text-4xl md:text-5xl font-serif text-zinc-900 leading-tight mb-4">
                      Opensource UI & Design Integration
                    </h1>
                    <p className="text-lg text-zinc-600 leading-relaxed">
                      Frontend engineering guide for developers contributing to KidneyAI, adhering strictly to the Opensource UI specification.
                    </p>
                  </div>

                  {/* Design Principles */}
                  <div className="space-y-4">
                    <h2 className="text-2xl font-serif font-bold text-zinc-900">
                      Core Design Principles
                    </h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="text-xs font-bold text-zinc-900 uppercase">1. Neutral Stage Surfaces</div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          We use pure white and near-white (<code className="font-mono">#FFFFFF</code>, <code className="font-mono">zinc-50</code>) surfaces paired with 1px neutral borders (<code className="font-mono">neutral-200</code>). Heavy drop shadows and saturated neon gradients are strictly avoided.
                        </p>
                      </div>

                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="text-xs font-bold text-zinc-900 uppercase">2. Strict Typography Hierarchy</div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          <strong>Instrument Serif</strong> is reserved for page titles and brand display headers. <strong>Inter / Geist</strong> is used for numerical readouts, body paragraphs, and clinical labels.
                        </p>
                      </div>

                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="text-xs font-bold text-zinc-900 uppercase">3. Clinical State Accents</div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Color is reserved exclusively for clinical semantics: <strong>Rose (alert/pathology)</strong> and <strong>Emerald (clear/normal)</strong>. All other controls remain neutral monochrome.
                        </p>
                      </div>

                      <div className="p-5 border border-neutral-200 rounded-2xl bg-white space-y-2">
                        <div className="text-xs font-bold text-zinc-900 uppercase">4. Tactile Spring Physics</div>
                        <p className="text-xs text-zinc-600 leading-relaxed">
                          Interactive elements and state transitions utilize <strong>Framer Motion</strong> spring dynamics (<code className="font-mono">stiffness: 300, damping: 25</code>) to give the application a physical, responsive feel.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Dropzone Code Snippet */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-zinc-500 font-mono uppercase text-[11px]">
                      <span>Tactile Dropzone Spring Implementation (React / Framer Motion)</span>
                      <button
                        onClick={() => handleCopy(`// Opensource UI Tactile Dropzone
<motion.div
  whileHover={{ scale: 1.01 }}
  whileTap={{ scale: 0.99 }}
  transition={{ type: "spring", stiffness: 300, damping: 25 }}
  className="border border-dashed border-neutral-300 rounded-2xl p-8 bg-zinc-50/50 hover:bg-zinc-50"
>
  <UploadCloud className="text-zinc-400" />
</motion.div>`, "react-dropzone")}
                        className="flex items-center gap-1 text-zinc-600 hover:text-zinc-900"
                      >
                        {copiedCode === "react-dropzone" ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                        <span>{copiedCode === "react-dropzone" ? "Copied" : "Copy Snippet"}</span>
                      </button>
                    </div>

                    <pre className="p-4 bg-zinc-900 text-zinc-100 rounded-2xl overflow-x-auto font-mono text-xs leading-relaxed">
{`import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";

export function TactileScanDropzone({ onFileSelect }: { onFileSelect: (f: File) => void }) {
  return (
    <motion.div
      whileHover={{ scale: 1.008 }}
      whileTap={{ scale: 0.992 }}
      transition={{ type: "spring", stiffness: 320, damping: 24 }}
      className="border border-dashed border-neutral-300 rounded-2xl p-8 bg-zinc-50/60 
                 hover:bg-zinc-50 hover:border-neutral-400 cursor-pointer transition-colors"
    >
      <div className="flex flex-col items-center justify-center text-center">
        <UploadCloud className="text-zinc-400 mb-3" size={32} />
        <h4 className="font-serif text-lg text-zinc-900 mb-1">Upload Renal Ultrasound Scan</h4>
        <p className="text-xs text-zinc-500">Drag & drop DICOM export, PNG, or JPEG</p>
      </div>
    </motion.div>
  );
}`}
                    </pre>
                  </div>

                  {/* Navigation Footer */}
                  <div className="pt-8 border-t border-neutral-200 flex justify-between">
                    <button
                      onClick={() => setActiveTab("fastapi")}
                      className="inline-flex items-center gap-2 px-4 py-2 border border-neutral-200 text-zinc-700 text-sm font-medium rounded-xl hover:bg-zinc-50 transition-colors"
                    >
                      <ArrowLeft size={16} />
                      <span>Previous: FastAPI Reference</span>
                    </button>
                    <Link
                      href="/"
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-600 text-white text-sm font-medium rounded-xl hover:bg-emerald-700 transition-colors shadow-xs"
                    >
                      <span>Open Clinical Scanner</span>
                      <ArrowLeft size={16} className="rotate-180" />
                    </Link>
                  </div>
                </div>
              )}

            </div>
          </main>
        </div>
      </div>
    </>
  );
}
