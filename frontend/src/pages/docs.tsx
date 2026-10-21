import React, { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { 
  ShieldCheck, BookOpen, Database, Github, ArrowLeft, 
  Activity, CheckCircle2, AlertTriangle, Cpu, Layers, 
  FileText, Terminal, BarChart2, ShieldAlert
} from "lucide-react";

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("overview");

  const scrollTo = (id: string) => {
    setActiveSection(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <>
      <Head>
        <title>Documentation | KidneyAI Clinical Platform</title>
        <meta name="description" content="Technical documentation, deep learning architecture, benchmarks, and API reference for KidneyAI." />
      </Head>

      <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans selection:bg-zinc-200 flex flex-col">
        {/* ==================== NAVBAR ==================== */}
        <header className="w-full border-b border-neutral-200 bg-white/80 backdrop-blur-md sticky top-0 z-20">
          <div className="max-w-6xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 cursor-pointer">
              <ShieldCheck className="text-emerald-600" size={24} />
              <span className="font-serif text-2xl tracking-tight mt-1">
                KidneyAI
              </span>
              <span className="ml-2 text-xs font-semibold px-2 py-0.5 bg-zinc-100 text-zinc-600 rounded-md border border-zinc-200">
                Docs
              </span>
            </Link>

            <nav className="flex items-center gap-5 text-sm font-medium text-zinc-500">
              <Link href="/" className="flex items-center gap-1.5 text-zinc-900 hover:text-emerald-600 transition-colors">
                <ArrowLeft size={16} /> Scanner
              </Link>
              <a 
                href="https://www.kaggle.com/datasets/safurahikmat/kidney-stone-ultrasound-image-dataset" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors"
              >
                <Database size={16} /> Dataset
              </a>
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors"
              >
                <Github size={16} /> GitHub
              </a>
            </nav>
          </div>
        </header>

        {/* ==================== CONTENT WORKSPACE ==================== */}
        <div className="max-w-6xl w-full mx-auto px-4 md:px-6 py-10 flex-1 grid grid-cols-1 md:grid-cols-[240px_1fr] gap-10">
          {/* Sidebar Navigation */}
          <aside className="hidden md:block sticky top-24 h-[calc(100vh-8rem)] overflow-y-auto pr-4 border-r border-neutral-200/80">
            <div className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-3">
              Documentation
            </div>
            <ul className="space-y-1 text-sm">
              {[
                { id: "overview", label: "Project Overview", icon: FileText },
                { id: "modality-guard", label: "Modality Guard (OOD)", icon: ShieldAlert },
                { id: "architecture", label: "Model Architecture", icon: Cpu },
                { id: "benchmarks", label: "Benchmarks & Accuracy", icon: BarChart2 },
                { id: "gradcam", label: "Grad-CAM Explainability", icon: Layers },
                { id: "dilation", label: "Pelvicalyceal Grading", icon: Activity },
                { id: "api", label: "FastAPI Reference", icon: Terminal },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <li key={item.id}>
                    <button
                      onClick={() => scrollTo(item.id)}
                      className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-sm transition-colors ${
                        activeSection === item.id 
                          ? "bg-zinc-200/70 text-zinc-900 font-semibold" 
                          : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
                      }`}
                    >
                      <Icon size={16} className={activeSection === item.id ? "text-emerald-600" : "text-zinc-400"} />
                      <span>{item.label}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </aside>

          {/* Main Documentation Articles */}
          <main className="space-y-14 max-w-3xl">
            {/* Section: Overview */}
            <section id="overview" className="scroll-mt-24">
              <span className="text-xs font-bold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 uppercase tracking-wide">
                Clinical AI System
              </span>
              <h1 className="text-4xl md:text-5xl font-serif tracking-tight text-zinc-900 mt-3 mb-4">
                NephroScan AI / KidneyAI
              </h1>
              <p className="text-lg text-zinc-600 leading-relaxed mb-6">
                KidneyAI is a deep learning clinical decision support system engineered for high-accuracy nephrolithiasis (kidney stone) detection and pelvicalyceal dilation assessment from B-mode ultrasound scans.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-6">
                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="text-xs text-zinc-400 font-medium uppercase">Test Accuracy</div>
                  <div className="text-3xl font-serif font-bold text-zinc-900 mt-1">98.87%</div>
                  <div className="text-xs text-emerald-600 font-medium mt-1">Evaluated on 1,676 scans</div>
                </div>
                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="text-xs text-zinc-400 font-medium uppercase">F1-Score</div>
                  <div className="text-3xl font-serif font-bold text-zinc-900 mt-1">0.99</div>
                  <div className="text-xs text-zinc-500 mt-1">Balanced calculus precision</div>
                </div>
                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="text-xs text-zinc-400 font-medium uppercase">Modality Guard</div>
                  <div className="text-3xl font-serif font-bold text-zinc-900 mt-1">Active</div>
                  <div className="text-xs text-zinc-500 mt-1">OOD rejection armed</div>
                </div>
              </div>
            </section>

            {/* Section: Modality Guard */}
            <section id="modality-guard" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <ShieldAlert className="text-rose-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  Ultrasound Modality Guard (OOD Prevention)
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                In standard binary classification networks (sigmoid activation: σ(W · x + b)), the model operates in a closed world: it assumes every input image is a kidney ultrasound. When given photos of animals (such as pandas), scenery, cars, or documents, high-contrast black-and-white patches trigger false positives with up to 100% confidence.
              </p>
              <div className="p-5 border border-amber-200 bg-amber-50/60 rounded-xl mb-6">
                <h4 className="text-sm font-bold text-amber-900 mb-1 flex items-center gap-1.5">
                  <AlertTriangle size={16} /> How KidneyAI Prevents Non-Medical False Positives
                </h4>
                <p className="text-xs text-amber-800 leading-relaxed">
                  Before running neural network inference, the <strong>Ultrasound Modality Guard</strong> analyzes 4 physical acoustic scan invariants:
                </p>
                <ul className="text-xs text-amber-900 list-disc list-inside mt-2 space-y-1">
                  <li><strong>Color Saturation Check:</strong> Real B-mode ultrasound is monochromatic. Saturation threshold &le; 0.18.</li>
                  <li><strong>RGB Channel Divergence:</strong> Acoustic scans have near-identical RGB channels (mean difference &le; 16.0).</li>
                  <li><strong>Speckle Noise Variance:</strong> Acoustic wave interference creates signature speckle textures (Laplacian variance &ge; 8.0).</li>
                  <li><strong>Dynamic Contrast & Intensity:</strong> Eliminates blank, overexposed, or pure uniform graphics.</li>
                </ul>
              </div>
            </section>

            {/* Section: Model Architecture */}
            <section id="architecture" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="text-emerald-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  Model Architecture & Transfer Learning
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                We employ <strong>MobileNetV2</strong> pre-trained on ImageNet as a feature extractor, with a lightweight custom classification head for optimal inference latency on clinical edge devices:
              </p>
              <div className="p-4 bg-zinc-900 text-zinc-100 rounded-xl font-mono text-xs overflow-x-auto leading-relaxed border border-zinc-800">
                Input Image (224 × 224 × 3, [0, 1] normalized)<br/>
                &nbsp;&nbsp;└── MobileNetV2 Base (ImageNet Weights, Frozen)<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── GlobalAveragePooling2D()<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Dropout(rate=0.3)<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Dense(128, activation=&apos;relu&apos;)<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Dropout(rate=0.2)<br/>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Dense(1, activation=&apos;sigmoid&apos;) → Calculus Probability [0.0 - 1.0]
              </div>
            </section>

            {/* Section: Benchmarks */}
            <section id="benchmarks" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <BarChart2 className="text-emerald-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  Independent Test Benchmarks
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                The model was benchmarked across <strong>1,676 unseen clinical test scans</strong> partitioned from the ultrasound dataset:
              </p>
              <div className="border border-neutral-200 rounded-xl overflow-hidden bg-white shadow-2xs">
                <table className="w-full text-left text-sm">
                  <thead className="bg-zinc-50 border-b border-neutral-200 text-xs font-semibold text-zinc-600">
                    <tr>
                      <th className="p-3">Class</th>
                      <th className="p-3">Precision</th>
                      <th className="p-3">Recall</th>
                      <th className="p-3">F1-Score</th>
                      <th className="p-3">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100 font-mono text-xs">
                    <tr>
                      <td className="p-3 font-sans font-medium text-zinc-900">Normal (No Stone)</td>
                      <td className="p-3 text-zinc-700">0.99</td>
                      <td className="p-3 text-zinc-700">0.99</td>
                      <td className="p-3 text-zinc-700">0.99</td>
                      <td className="p-3 text-zinc-500">838</td>
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-medium text-zinc-900">Calculus (Stone)</td>
                      <td className="p-3 text-zinc-700">0.99</td>
                      <td className="p-3 text-zinc-700">0.98</td>
                      <td className="p-3 text-zinc-700">0.99</td>
                      <td className="p-3 text-zinc-500">838</td>
                    </tr>
                    <tr className="bg-zinc-50/50 font-semibold font-sans">
                      <td className="p-3">Weighted Average</td>
                      <td className="p-3 font-mono">0.99</td>
                      <td className="p-3 font-mono">0.98</td>
                      <td className="p-3 font-mono">0.99</td>
                      <td className="p-3 font-mono text-zinc-500">1,676</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Section: Grad-CAM */}
            <section id="gradcam" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <Layers className="text-emerald-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  Explainable AI (Grad-CAM Activation Maps)
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                To guarantee diagnostic transparency and clinical explainability, KidneyAI computes Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the <code>out_relu</code> convolutional layer of MobileNetV2.
              </p>
              <p className="text-zinc-600 leading-relaxed">
                The gradient of the calculus score with respect to convolutional feature activation maps highlights the precise spatial region of hyperechoic acoustic reflection, allowing clinicians to immediately verify whether the AI focused on an authentic calculus or artifact.
              </p>
            </section>

            {/* Section: Pelvicalyceal Grading */}
            <section id="dilation" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="text-emerald-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  Pelvicalyceal Dilation & Severity Index
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                The system correlates calculus probability and focal stone activation area to estimate potential pelvicalyceal separation and obstructive hydronephrosis risk:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div className="p-4 border border-neutral-200 rounded-xl bg-white">
                  <div className="font-bold text-zinc-900 mb-1">Normal / Low Risk</div>
                  <div className="text-zinc-500">Probability &lt; 0.50. Baseline normal renal parenchyma with unremarkable architecture.</div>
                </div>
                <div className="p-4 border border-amber-200 rounded-xl bg-amber-50/50">
                  <div className="font-bold text-amber-900 mb-1">Mild Dilation</div>
                  <div className="text-amber-800">Probability 0.50 - 0.85. Possible early subacute obstruction (~50% hydronephrosis risk).</div>
                </div>
                <div className="p-4 border border-rose-200 rounded-xl bg-rose-50/50">
                  <div className="font-bold text-rose-900 mb-1">Moderate-to-Severe</div>
                  <div className="text-rose-800">Probability &gt; 0.85. Obstructive calculus with marked acoustic shadowing (&gt;75% risk).</div>
                </div>
              </div>
            </section>

            {/* Section: API Reference */}
            <section id="api" className="scroll-mt-24 border-t border-neutral-200 pt-10">
              <div className="flex items-center gap-2 mb-2">
                <Terminal className="text-emerald-600" size={20} />
                <h2 className="text-2xl font-serif font-bold text-zinc-900">
                  FastAPI REST Endpoints
                </h2>
              </div>
              <p className="text-zinc-600 leading-relaxed mb-4">
                The backend service provides high-performance asynchronous REST endpoints:
              </p>

              <div className="space-y-4">
                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-bold">POST</span>
                    <span className="font-semibold text-zinc-900">/api/v1/predict</span>
                  </div>
                  <p className="text-xs text-zinc-500 mt-2">
                    Takes a multipart ultrasound image file and optional <code>validate_scan</code> flag. Returns diagnosis, confidence score, Grad-CAM Base64 data URI, and pelvicalyceal dilation metrics.
                  </p>
                </div>

                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-800 rounded font-bold">POST</span>
                    <span className="font-semibold text-zinc-900">/api/v1/validate</span>
                  </div>
                  <p className="text-xs text-zinc-500 mt-2">
                    Lightweight pre-inference validation checking whether an image meets B-mode medical ultrasound criteria.
                  </p>
                </div>

                <div className="p-4 border border-neutral-200 rounded-xl bg-white shadow-2xs">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="px-2 py-0.5 bg-zinc-100 text-zinc-800 rounded font-bold">GET</span>
                    <span className="font-semibold text-zinc-900">/api/v1/health</span>
                  </div>
                  <p className="text-xs text-zinc-500 mt-2">
                    Readiness probe returning model loading status and API version.
                  </p>
                </div>
              </div>
            </section>
          </main>
        </div>
      </div>
    </>
  );
}
