import React, { useState } from "react";
import { 
  FileText, ShieldAlert, Cpu, BarChart, Layers, 
  Activity, Code, ShieldCheck, Database, Github, 
  ArrowLeft, AlertTriangle, CheckCircle2
} from "lucide-react";

export default function App() {
  const [activePage, setActivePage] = useState("dataset");

  const sidebarLinks = [
    { id: "overview", icon: FileText, label: "Project Overview" },
    { id: "modality-guard", icon: ShieldAlert, label: "Modality Guard (OOD)" },
    { id: "dataset", icon: Database, label: "Training Data & Hygiene" },
    { id: "architecture", icon: Cpu, label: "Model Architecture" },
    { id: "grad-cam", icon: Layers, label: "Grad-CAM Explainability" },
    { id: "fastapi", icon: Code, label: "FastAPI Reference" },
  ];

  return (
    <>
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');
        
        .font-serif { font-family: 'Instrument Serif', serif; }
        .font-sans { font-family: 'Inter', sans-serif; }
      `}} />

      <div className="min-h-screen bg-white text-zinc-900 font-sans flex flex-col selection:bg-zinc-200">
        
        {/* ==================== SECTION: TOP NAVIGATION ==================== */}
        <header className="w-full border-b border-neutral-200 bg-white sticky top-0 z-20 h-16 flex items-center justify-between px-4 md:px-6">
          <div className="flex items-center gap-3">
            <ShieldCheck className="text-emerald-600" size={24} />
            <span className="font-serif text-2xl tracking-tight mt-1">
              KidneyAI
            </span>
            <span className="hidden sm:inline-block ml-2 px-2 py-0.5 rounded-md bg-zinc-100 border border-neutral-200 text-xs font-medium text-zinc-600">
              Docs
            </span>
          </div>
          
          <nav className="flex items-center gap-4 sm:gap-6 text-sm font-medium text-zinc-500">
            <a href="#scanner" className="flex items-center gap-2 hover:text-zinc-900 transition-colors">
              <ArrowLeft size={16} /> <span className="hidden sm:inline">Scanner</span>
            </a>
            <a href="https://github.com/bappaditya-paul/kidney-stone-detection-aiml" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-zinc-900 transition-colors">
              <Github size={16} /> <span className="hidden sm:inline">GitHub</span>
            </a>
          </nav>
        </header>

        {/* ==================== SECTION: MAIN LAYOUT ==================== */}
        <div className="flex flex-1 overflow-hidden">
          
          {/* LEFT SIDEBAR */}
          <aside className="w-64 border-r border-neutral-200 bg-zinc-50 flex-shrink-0 hidden md:block overflow-y-auto">
            <div className="p-4">
              <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-4 px-3">
                Documentation
              </h3>
              <nav className="space-y-1">
                {sidebarLinks.map((link) => {
                  const isActive = activePage === link.id;
                  const Icon = link.icon;
                  return (
                    <button
                      key={link.id}
                      onClick={() => setActivePage(link.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors text-left ${
                        isActive 
                          ? "bg-white border border-neutral-200 text-zinc-900 font-medium shadow-sm" 
                          : "border border-transparent text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
                      }`}
                    >
                      <Icon size={16} className={isActive ? "text-zinc-700" : "text-zinc-400"} />
                      {link.label}
                    </button>
                  );
                })}
              </nav>
            </div>
          </aside>

          {/* RIGHT CONTENT AREA */}
          <main className="flex-1 overflow-y-auto bg-white">
            <div className="max-w-4xl mx-auto py-10 px-6 md:py-12 md:px-10">
              
              {/* Page Title */}
              <div className="mb-10">
                <span className="inline-flex items-center text-xs font-semibold tracking-wide text-zinc-600 bg-zinc-100 border border-neutral-200 px-2 py-1 rounded-md uppercase mb-4">
                  Dataset & Methodology
                </span>
                <h1 className="text-4xl md:text-5xl text-zinc-900 mb-6 leading-tight font-serif">
                  Training Data & Hygiene
                </h1>
                <p className="text-base md:text-lg text-zinc-600 leading-relaxed">
                  The KidneyAI model is trained on a highly curated, open-source Kaggle dataset of B-mode renal ultrasound scans. Strict dataset hygiene protocols ensure zero data leakage and high clinical validity.
                </p>
              </div>

              {/* Quick Stats / Kaggle Link */}
              <div className="flex flex-wrap items-center gap-4 mb-10">
                <a href="https://www.kaggle.com/datasets/jayaprakashpondy/kidney-cancer-image" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-900 text-white text-sm font-medium rounded-md hover:bg-zinc-800 transition-colors border border-transparent">
                  <Database size={16} />
                  View Dataset on Kaggle
                </a>
                <span className="text-sm text-zinc-500 flex items-center gap-1.5 font-medium">
                  <CheckCircle2 size={16} className="text-emerald-500" /> 
                  V2 Curated Release
                </span>
              </div>

              {/* Clinical Warning Box */}
              <div className="my-8 bg-amber-50 border border-amber-200 rounded-xl p-6">
                <h4 className="flex items-center gap-2 text-amber-900 font-semibold mb-3 text-sm uppercase tracking-wide">
                  <AlertTriangle size={18} />
                  Critical Context: Dataset Hygiene
                </h4>
                <p className="text-sm text-amber-800 mb-4 leading-relaxed">
                  In medical AI, data leakage (where train and test sets contain images from the same patient) artificially inflates accuracy. To prevent this:
                </p>
                <ul className="space-y-2 text-sm text-amber-800 list-disc pl-5 marker:text-amber-400">
                  <li>Automated SHA-256 hash deduplication was performed to eliminate duplicate scans.</li>
                  <li>Corrupted or pure-black DICOM frames were purged using intensity variance thresholds.</li>
                  <li>A strict holdout partition (1,676 scans) was frozen prior to hyperparameter tuning.</li>
                </ul>
              </div>

              {/* Dataset Composition Table */}
              <h3 className="text-2xl text-zinc-900 mt-12 mb-6 font-serif">
                Dataset Composition
              </h3>
              
              <div className="border border-neutral-200 rounded-xl overflow-hidden mb-12">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm whitespace-nowrap">
                    <thead className="bg-zinc-50 border-b border-neutral-200">
                      <tr>
                        <th className="px-6 py-4 font-medium text-zinc-600">Classification</th>
                        <th className="px-6 py-4 font-medium text-zinc-600">Training Split (80%)</th>
                        <th className="px-6 py-4 font-medium text-zinc-600">Test Split (20%)</th>
                        <th className="px-6 py-4 font-medium text-zinc-600">Total Scans</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-200">
                      <tr className="hover:bg-zinc-50 transition-colors">
                        <td className="px-6 py-4 font-medium text-zinc-900 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Normal (Class 0)
                        </td>
                        <td className="px-6 py-4 text-zinc-600">10,480</td>
                        <td className="px-6 py-4 text-zinc-600">838</td>
                        <td className="px-6 py-4 font-medium text-zinc-900">11,318</td>
                      </tr>
                      <tr className="hover:bg-zinc-50 transition-colors">
                        <td className="px-6 py-4 font-medium text-zinc-900 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-rose-500"></span> Calculus (Class 1)
                        </td>
                        <td className="px-6 py-4 text-zinc-600">10,480</td>
                        <td className="px-6 py-4 text-zinc-600">838</td>
                        <td className="px-6 py-4 font-medium text-zinc-900">11,318</td>
                      </tr>
                      <tr className="bg-zinc-50">
                        <td className="px-6 py-4 font-medium text-zinc-900">Total</td>
                        <td className="px-6 py-4 font-medium text-zinc-900">20,960</td>
                        <td className="px-6 py-4 font-medium text-zinc-900">1,676</td>
                        <td className="px-6 py-4 font-medium text-zinc-900">22,636</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          </main>
        </div>
      </div>
    </>
  );
}