import React, { useState } from "react";
import { motion, type Variants } from "framer-motion";
import {
  FileText, ShieldAlert, Cpu, Layers, Activity,
  Code, ShieldCheck, Database, Github, ExternalLink,
  Copy, Check, Terminal, CheckCircle2, AlertTriangle,
  ArrowRight, BookOpen, Sparkles
} from "lucide-react";

const staggerContainer: Variants = {
  animate: { transition: { staggerChildren: 0.08 } }
};

const fadeUp: Variants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } }
};

export const DocsTab: React.FC = () => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <motion.div
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      className="w-full max-w-5xl mx-auto py-10 md:py-16 px-4 sm:px-6 text-zinc-900 pb-28"
    >
      {/* Hero Header */}
      <motion.div variants={fadeUp} className="mb-12 md:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-white border border-zinc-200/90 rounded-full text-xs font-semibold uppercase tracking-wider text-zinc-600 mb-6 shadow-2xs">
          <BookOpen size={14} className="text-emerald-600" />
          <span>GeeksforGeeks-Style Architecture & Clinical Guide</span>
        </div>
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-serif text-zinc-900 mb-4 tracking-tight leading-tight">
          How KidneyAI Was Built: From Physics to Production
        </h1>
        <p className="text-base sm:text-lg text-zinc-600 max-w-3xl font-light leading-relaxed">
          A first-principles engineering and deep learning breakdown of NephroScan AI (KidneyAI).
          Learn how acoustic impedance physics, Out-of-Distribution modality guards, MobileNetV2
          bottlenecks, and Grad-CAM explainability unite to detect renal calculi with 98.87% accuracy.
        </p>

        {/* Important Model Limitation Banner */}
        <div className="mt-8 bg-amber-500/10 border border-amber-500/30 rounded-3xl p-5 sm:p-6 text-zinc-800 shadow-2xs">
          <div className="flex items-start gap-3.5">
            <div className="w-9 h-9 rounded-xl bg-amber-500/20 text-amber-700 flex items-center justify-center shrink-0 mt-0.5">
              <AlertTriangle size={20} />
            </div>
            <div className="space-y-1.5 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h4 className="text-sm font-semibold text-zinc-900">
                  Critical Scope & Diagnostic Limitations
                </h4>
                <span className="px-2 py-0.5 rounded-full bg-amber-200/80 text-amber-900 text-[10px] font-mono font-medium">
                  Binary Classifier Only (Stone vs. No Stone)
                </span>
                <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 text-[10px] font-mono font-medium">
                  Not a General Organ Scanner
                </span>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed font-light">
                NephroScan AI is trained <strong>strictly on two classes: Healthy Ultrasound vs. Kidney Stone Ultrasound</strong>. 
                Because deep learning models operate under a <em>closed-world assumption</em>, the model <strong>cannot identify unknown objects</strong>, cysts, tumors, or other organs. Feeding it dark-mode laptop/terminal screenshots will cause false positive stone detections because white text characters mimic hyperechoic calculus reflections.
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* GitHub & Kaggle Repository Cards (Prominently placed in Docs) */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-16">
        {/* GitHub Card */}
        <div className="bg-white rounded-3xl p-6 border border-zinc-200/90 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-zinc-900 text-white flex items-center justify-center">
                <Github size={20} />
              </div>
              <div>
                <h4 className="font-semibold text-zinc-900 text-base">GitHub Repository</h4>
                <p className="text-xs text-zinc-500 font-mono">bappaditya-paul/kidney-stone-detection-aiml</p>
              </div>
            </div>
            <a
              href="https://github.com/bappaditya-paul/kidney-stone-detection-aiml"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100 transition-colors"
              title="Open GitHub Repository"
            >
              <ExternalLink size={18} />
            </a>
          </div>

          <p className="text-xs text-zinc-600 leading-relaxed mb-4 font-light">
            Complete production codebase containing FastAPI microservices, pipeline training scripts,
            and Next.js frontend code ready for local or cloud deployment.
          </p>

          <div className="flex items-center justify-between bg-zinc-50 rounded-xl p-2.5 border border-zinc-200/70 font-mono text-xs text-zinc-700">
            <span className="truncate pr-2">git clone https://github.com/bappaditya-paul/kidney-stone-detection-aiml.git</span>
            <button
              onClick={() =>
                copyToClipboard(
                  "git clone https://github.com/bappaditya-paul/kidney-stone-detection-aiml.git",
                  "git-clone"
                )
              }
              className="p-1.5 hover:bg-zinc-200 rounded-md text-zinc-600 transition-colors shrink-0"
              title="Copy clone command"
            >
              {copiedId === "git-clone" ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
            </button>
          </div>
        </div>

        {/* Kaggle Dataset Card */}
        <div className="bg-white rounded-3xl p-6 border border-zinc-200/90 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200/80 flex items-center justify-center">
                <Database size={20} />
              </div>
              <div>
                <h4 className="font-semibold text-zinc-900 text-base">Kaggle Dataset</h4>
                <p className="text-xs text-zinc-500 font-mono">jayaprakashpondy/kidney-cancer-image</p>
              </div>
            </div>
            <a
              href="https://www.kaggle.com/datasets/jayaprakashpondy/kidney-cancer-image"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100 transition-colors"
              title="Open Kaggle Dataset"
            >
              <ExternalLink size={18} />
            </a>
          </div>

          <p className="text-xs text-zinc-600 leading-relaxed mb-4 font-light">
            Standardized B-mode ultrasound and CT cross-correlation dataset with 1,676 strict
            holdout test scans, rigorously balanced across calculus and normal categories.
          </p>

          <div className="flex items-center justify-between bg-zinc-50 rounded-xl p-2.5 border border-zinc-200/70 font-mono text-xs text-zinc-700">
            <span className="truncate pr-2">kaggle datasets download -d jayaprakashpondy/kidney-cancer-image</span>
            <button
              onClick={() =>
                copyToClipboard(
                  "kaggle datasets download -d jayaprakashpondy/kidney-cancer-image",
                  "kaggle-download"
                )
              }
              className="p-1.5 hover:bg-zinc-200 rounded-md text-zinc-600 transition-colors shrink-0"
              title="Copy Kaggle command"
            >
              {copiedId === "kaggle-download" ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
            </button>
          </div>
        </div>
      </motion.div>

      {/* SECTION 1: Sonographic Physics & Clinical Background */}
      <motion.section variants={fadeUp} className="mb-16">
        <div className="flex items-center gap-2 mb-2 text-emerald-700 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>01</span>
          <span>•</span>
          <span>Acoustic Physics & Clinical Pathology</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          1. The Clinical Problem & Ultrasound Physics
        </h2>

        <div className="prose prose-zinc max-w-none text-zinc-700 text-sm sm:text-base leading-relaxed space-y-4 font-light mb-8">
          <p>
            <strong>Nephrolithiasis (Renal Calculi)</strong> impacts approximately 10% to 12% of the
            global population, with a 50% recurrence rate within 5 years. When a stone obstructs the
            renal pelvis or ureteropelvic junction, upstream hydrostatic pressure creates
            <strong> secondary hydronephrosis</strong> (dilation of the pelvicalyceal collecting system),
            risking acute kidney injury and parenchymal atrophy.
          </p>
          <p>
            While non-contrast CT (NCCT KUB) is sensitive, it delivers 3 to 5 mSv of ionizing radiation.
            The European Association of Urology (EAU) recommends <strong>B-mode renal ultrasound</strong>
            as the primary first-line modality because it involves <strong>zero radiation</strong>, is
            inexpensive, and is deployable bedside at point-of-care (POCUS).
          </p>
        </div>

        {/* Sonographic Physics Visual Figure Comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Normal Kidney Figure */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 overflow-hidden shadow-2xs">
            <div className="aspect-[16/10] bg-zinc-950 p-6 flex flex-col items-center justify-center relative overflow-hidden">
              <svg width="240" height="150" viewBox="0 0 240 150" fill="none" className="opacity-90">
                <path d="M40 25 Q 120 10 200 35 Q 220 85 180 125 Q 120 140 60 120 Q 20 80 40 25 Z" fill="#18181b" stroke="#3f3f46" strokeWidth="2" />
                <ellipse cx="120" cy="75" rx="45" ry="35" fill="#27272a" />
                <ellipse cx="120" cy="75" rx="18" ry="14" fill="#3f3f46" opacity="0.6" />
                <text x="12" y="20" fill="#a1a1aa" fontFamily="monospace" fontSize="10">UNIFORM CORTEX</text>
                <text x="75" y="145" fill="#34d399" fontFamily="monospace" fontSize="10">NO SHADOWING</text>
              </svg>
              <div className="absolute top-3 right-3 px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono text-[10px] border border-emerald-500/30">
                PHYSIOLOGICAL
              </div>
            </div>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-2 flex items-center gap-2">
                <CheckCircle2 size={18} className="text-emerald-600" />
                Figure 1: Healthy Renal Architecture
              </h3>
              <p className="text-xs sm:text-sm text-zinc-600 leading-relaxed font-light">
                Displays a compact central echo complex (sinus fat and pelvicalyceal system).
                Corticomedullary differentiation is crisp, without hyperechoic calcifications or
                posterior acoustic shadowing.
              </p>
            </div>
          </div>

          {/* Calculus Kidney Figure */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 overflow-hidden shadow-2xs">
            <div className="aspect-[16/10] bg-zinc-950 p-6 flex flex-col items-center justify-center relative overflow-hidden">
              <svg width="240" height="150" viewBox="0 0 240 150" fill="none" className="opacity-90">
                <path d="M40 25 Q 120 10 200 35 Q 220 85 180 125 Q 120 140 60 120 Q 20 80 40 25 Z" fill="#18181b" stroke="#3f3f46" strokeWidth="2" />
                <ellipse cx="120" cy="75" rx="45" ry="35" fill="#27272a" />
                {/* Stone */}
                <circle cx="132" cy="62" r="7" fill="#ffffff" filter="drop-shadow(0px 0px 4px #ffffff)" />
                {/* Acoustic Shadow */}
                <path d="M125 69 L115 145 L149 145 Z" fill="#000000" opacity="0.9" />
                <text x="145" y="65" fill="#fb7185" fontFamily="monospace" fontSize="10">HYPERECHOIC</text>
                <text x="80" y="145" fill="#fb7185" fontFamily="monospace" fontSize="10">ACOUSTIC SHADOW</text>
              </svg>
              <div className="absolute top-3 right-3 px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-mono text-[10px] border border-rose-500/30">
                CALCULUS
              </div>
            </div>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-zinc-900 mb-2 flex items-center gap-2">
                <ShieldAlert size={18} className="text-rose-600" />
                Figure 2: Renal Calculus with Acoustic Shadow
              </h3>
              <p className="text-xs sm:text-sm text-zinc-600 leading-relaxed font-light">
                Marked acoustic impedance mismatch ($Z = \rho \cdot v$). The mineralized stone produces
                intense reflection (hyperechoic white focus) followed by a distal acoustic void (dark shadow band).
              </p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* SECTION 2: The 4-Tier KidneyAI Neural Pipeline */}
      <motion.section variants={fadeUp} className="mb-16">
        <div className="flex items-center gap-2 mb-2 text-emerald-700 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>02</span>
          <span>•</span>
          <span>System Architecture</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          2. The 4-Tier Defensive Neural Pipeline
        </h2>

        <p className="text-sm sm:text-base text-zinc-700 font-light leading-relaxed mb-8">
          Standard deep learning models fail in clinical deployment due to the <em>closed-world assumption</em>:
          a naive CNN will blindly classify a photo of a panda, car, or random selfie as 100% "Kidney Stone"
          because intermediate edge filters get stimulated by high-contrast fur or metal. KidneyAI solves this
          with an uncompromising four-tier defensive architecture:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs hover:-translate-y-1 transition-transform">
            <div className="w-12 h-12 bg-amber-50 text-amber-700 border border-amber-200/80 rounded-2xl flex items-center justify-center mb-4">
              <ShieldAlert size={22} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-1.5 text-base">Tier 1: Modality Guard</h4>
            <p className="text-xs text-zinc-500 font-light leading-relaxed">
              Physics-based Out-of-Distribution filter. Rejects color photos and non-ultrasound imagery via
              grayscale chromatic purity, sector geometry, and Rayleigh speckle checks.
            </p>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs hover:-translate-y-1 transition-transform">
            <div className="w-12 h-12 bg-blue-50 text-blue-700 border border-blue-200/80 rounded-2xl flex items-center justify-center mb-4">
              <Cpu size={22} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-1.5 text-base">Tier 2: MobileNetV2</h4>
            <p className="text-xs text-zinc-500 font-light leading-relaxed">
              Inverted Residual Bottlenecks with depthwise separable convolutions. Delivers clinical accuracy
              in &lt;30ms, ideal for point-of-care mobile sonography.
            </p>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs hover:-translate-y-1 transition-transform">
            <div className="w-12 h-12 bg-purple-50 text-purple-700 border border-purple-200/80 rounded-2xl flex items-center justify-center mb-4">
              <Layers size={22} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-1.5 text-base">Tier 3: Grad-CAM XAI</h4>
            <p className="text-xs text-zinc-500 font-light leading-relaxed">
              Computes feature map gradients with respect to the calculus score. Projects heatmaps directly
              onto the scan so clinicians can visually verify stone localization.
            </p>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs hover:-translate-y-1 transition-transform">
            <div className="w-12 h-12 bg-emerald-50 text-emerald-700 border border-emerald-200/80 rounded-2xl flex items-center justify-center mb-4">
              <Activity size={22} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-1.5 text-base">Tier 4: Dilation Engine</h4>
            <p className="text-xs text-zinc-500 font-light leading-relaxed">
              Evaluates pelvicalyceal dilation millimeter index and secondary hydronephrosis risk to determine
              whether surgical stenting or hydration is warranted.
            </p>
          </div>
        </div>
      </motion.section>

      {/* SECTION 3: Mathematical Foundations */}
      <motion.section variants={fadeUp} className="mb-16">
        <div className="flex items-center gap-2 mb-2 text-emerald-700 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>03</span>
          <span>•</span>
          <span>Mathematics & Formulations</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          3. Mathematical & Algorithmic Foundations
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Math Card 1: Acoustic Impedance & Reflection */}
          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs">
            <h4 className="text-base font-semibold text-zinc-900 mb-2">
              Acoustic Impedance & Reflection Coefficient
            </h4>
            <p className="text-xs text-zinc-600 mb-4 leading-relaxed font-light">
              Acoustic impedance Z is defined as density &rho; multiplied by sound speed v. At the interface
              between renal parenchyma (Z₁ &approx; 1.63 &times; 10⁶ Rayls) and
              calcium oxalate stone (Z₂ &approx; 7.5 &times; 10⁶ Rayls), the reflection
              power coefficient R is:
            </p>
            <div className="bg-zinc-900 text-emerald-400 p-4 rounded-2xl font-mono text-xs overflow-x-auto">
              {`R = ((Z2 - Z1) / (Z2 + Z1))^2 ≈ 0.41 (41% direct back-reflection)`}
            </div>
            <p className="text-[11px] text-zinc-500 mt-3 font-light">
              This high reflection coefficient creates the bright hyperechoic front and total distal shadow void.
            </p>
          </div>

          {/* Math Card 2: Grad-CAM Formulation */}
          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs">
            <h4 className="text-base font-semibold text-zinc-900 mb-2">
              Grad-CAM Neuron Activation Weighting
            </h4>
            <p className="text-xs text-zinc-600 mb-4 leading-relaxed font-light">
              Neuron importance weights $\alpha_k^c$ for class $c$ (Stone) and feature map $A^k$ of the final
              convolutional layer are computed via global average pooling of gradients:
            </p>
            <div className="bg-zinc-900 text-emerald-400 p-4 rounded-2xl font-mono text-xs overflow-x-auto">
              {`α_k^c = (1 / Z) * Σ_i Σ_j (∂y^c / ∂A_{i,j}^k)`}
              <br />
              {`L_{Grad-CAM}^c = ReLU( Σ_k α_k^c * A^k )`}
            </div>
            <p className="text-[11px] text-zinc-500 mt-3 font-light">
              The rectifier linear unit (ReLU) isolates features that have a positive influence on calculus detection.
            </p>
          </div>
        </div>
      </motion.section>

      {/* SECTION 4: Empirical Benchmarks & Validation */}
      <motion.section variants={fadeUp} className="mb-16">
        <div className="flex items-center gap-2 mb-2 text-emerald-700 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>04</span>
          <span>•</span>
          <span>Empirical Holdout Benchmarks</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          4. Empirical Validation on Holdout Scans
        </h2>

        {/* Big Numbers Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-zinc-900 text-white p-6 rounded-3xl flex flex-col items-center justify-center shadow-md">
            <span className="text-5xl font-serif text-emerald-400 mb-1">98.87%</span>
            <span className="text-[11px] uppercase tracking-widest font-semibold text-zinc-400">Holdout Accuracy</span>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 flex flex-col items-center justify-center shadow-2xs">
            <span className="text-5xl font-serif text-zinc-900 mb-1">0.99</span>
            <span className="text-[11px] uppercase tracking-widest font-semibold text-zinc-500">Precision (PPV)</span>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 flex flex-col items-center justify-center shadow-2xs">
            <span className="text-5xl font-serif text-zinc-900 mb-1">0.98</span>
            <span className="text-[11px] uppercase tracking-widest font-semibold text-zinc-500">Recall (Sensitivity)</span>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 flex flex-col items-center justify-center shadow-2xs">
            <span className="text-5xl font-serif text-zinc-900 mb-1">&lt;28ms</span>
            <span className="text-[11px] uppercase tracking-widest font-semibold text-zinc-500">Inference Latency</span>
          </div>
        </div>

        {/* Confusion Matrix Table */}
        <div className="bg-white rounded-3xl border border-zinc-200/90 p-6 shadow-2xs">
          <h4 className="text-base font-semibold text-zinc-900 mb-4 flex items-center gap-2">
            <Database size={18} className="text-emerald-600" />
            Independent Holdout Confusion Matrix (N = 1,676 Scans)
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs sm:text-sm text-left">
              <thead>
                <tr className="border-b border-zinc-200 text-zinc-500 uppercase tracking-wider text-[11px]">
                  <th className="py-3 px-4">Actual Condition</th>
                  <th className="py-3 px-4 text-center">Predicted: Normal</th>
                  <th className="py-3 px-4 text-center">Predicted: Calculus (Stone)</th>
                  <th className="py-3 px-4 text-right">Class Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 font-mono">
                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-zinc-800">True Normal (N = 838)</td>
                  <td className="py-3 px-4 text-center text-emerald-700 bg-emerald-50/50 font-semibold">829 (TN)</td>
                  <td className="py-3 px-4 text-center text-rose-600">9 (FP)</td>
                  <td className="py-3 px-4 text-right font-sans font-semibold text-zinc-900">98.92%</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-zinc-800">True Calculus (N = 838)</td>
                  <td className="py-3 px-4 text-center text-rose-600">10 (FN)</td>
                  <td className="py-3 px-4 text-center text-emerald-700 bg-emerald-50/50 font-semibold">828 (TP)</td>
                  <td className="py-3 px-4 text-right font-sans font-semibold text-zinc-900">98.81%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-zinc-500 mt-4 font-light">
            Patient-level split isolation guaranteed: no ultrasound slices from the same individual exist
            across both training and holdout validation sets.
          </p>
        </div>
      </motion.section>

      {/* SECTION 5: Step-by-Step "Build It Yourself" Implementation Guide */}
      <motion.section variants={fadeUp}>
        <div className="flex items-center gap-2 mb-2 text-emerald-700 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>05</span>
          <span>•</span>
          <span>Step-by-Step Reproduction</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          5. Step-by-Step Implementation Guide
        </h2>

        <p className="text-sm sm:text-base text-zinc-700 font-light leading-relaxed mb-8">
          Follow these sequential steps to train, run, and deploy the entire NephroScan AI stack
          from scratch on your local workstation or server:
        </p>

        <div className="space-y-6">
          {/* Step 1 */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 p-6 shadow-2xs">
            <div className="flex items-center gap-3 mb-3">
              <span className="w-7 h-7 rounded-full bg-zinc-900 text-white text-xs font-mono flex items-center justify-center font-bold">
                1
              </span>
              <h4 className="font-semibold text-zinc-900 text-base">Clone & Setup Python Environment</h4>
            </div>
            <p className="text-xs text-zinc-600 mb-3 font-light">
              Create an isolated virtual environment and install PyTorch, FastAPI, OpenCV, and dependencies:
            </p>
            <div className="bg-zinc-900 text-zinc-100 p-4 rounded-2xl font-mono text-xs overflow-x-auto relative group">
              <pre>{`git clone https://github.com/bappaditya-paul/kidney-stone-detection-aiml.git
cd kidney-stone-detection-aiml
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt`}</pre>
              <button
                onClick={() =>
                  copyToClipboard(
                    "git clone https://github.com/bappaditya-paul/kidney-stone-detection-aiml.git\ncd kidney-stone-detection-aiml\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt",
                    "step1"
                  )
                }
                className="absolute top-3 right-3 p-1.5 bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-colors"
                title="Copy commands"
              >
                {copiedId === "step1" ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>
          </div>

          {/* Step 2 */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 p-6 shadow-2xs">
            <div className="flex items-center gap-3 mb-3">
              <span className="w-7 h-7 rounded-full bg-zinc-900 text-white text-xs font-mono flex items-center justify-center font-bold">
                2
              </span>
              <h4 className="font-semibold text-zinc-900 text-base">Train MobileNetV2 Pipeline</h4>
            </div>
            <p className="text-xs text-zinc-600 mb-3 font-light">
              Execute the complete pipeline with acoustic augmentation and holdout validation:
            </p>
            <div className="bg-zinc-900 text-zinc-100 p-4 rounded-2xl font-mono text-xs overflow-x-auto relative group">
              <pre>{`python ml_pipeline/run_pipeline.py`}</pre>
              <button
                onClick={() => copyToClipboard("python ml_pipeline/run_pipeline.py", "step2")}
                className="absolute top-3 right-3 p-1.5 bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-colors"
                title="Copy commands"
              >
                {copiedId === "step2" ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>
          </div>

          {/* Step 3 */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 p-6 shadow-2xs">
            <div className="flex items-center gap-3 mb-3">
              <span className="w-7 h-7 rounded-full bg-zinc-900 text-white text-xs font-mono flex items-center justify-center font-bold">
                3
              </span>
              <h4 className="font-semibold text-zinc-900 text-base">Launch FastAPI Microservice</h4>
            </div>
            <p className="text-xs text-zinc-600 mb-3 font-light">
              Start the asynchronous REST API providing Modality Guard, inference, and Grad-CAM generation:
            </p>
            <div className="bg-zinc-900 text-zinc-100 p-4 rounded-2xl font-mono text-xs overflow-x-auto relative group">
              <pre>{`uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload`}</pre>
              <button
                onClick={() => copyToClipboard("uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload", "step3")}
                className="absolute top-3 right-3 p-1.5 bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-colors"
                title="Copy commands"
              >
                {copiedId === "step3" ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>
          </div>

          {/* Step 4 */}
          <div className="bg-white rounded-3xl border border-zinc-200/90 p-6 shadow-2xs">
            <div className="flex items-center gap-3 mb-3">
              <span className="w-7 h-7 rounded-full bg-zinc-900 text-white text-xs font-mono flex items-center justify-center font-bold">
                4
              </span>
              <h4 className="font-semibold text-zinc-900 text-base">Launch Next.js Clinical Cockpit</h4>
            </div>
            <p className="text-xs text-zinc-600 mb-3 font-light">
              In a separate terminal, launch the Next.js development server:
            </p>
            <div className="bg-zinc-900 text-zinc-100 p-4 rounded-2xl font-mono text-xs overflow-x-auto relative group">
              <pre>{`cd frontend
npm install
npm run dev`}</pre>
              <button
                onClick={() => copyToClipboard("cd frontend\nnpm install\nnpm run dev", "step4")}
                className="absolute top-3 right-3 p-1.5 bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-colors"
                title="Copy commands"
              >
                {copiedId === "step4" ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>
            <p className="text-xs text-zinc-500 mt-3 font-light">
              Access the clinical dashboard at <code className="bg-zinc-100 px-1.5 py-0.5 rounded text-zinc-800 font-mono">http://localhost:3000</code>.
            </p>
          </div>
        </div>
      </motion.section>

      {/* SECTION 6: Model Limitations & Out-Of-Distribution (OOD) Behavior */}
      <motion.section variants={fadeUp} className="mb-16">
        <div className="flex items-center gap-2 mb-2 text-rose-600 font-mono text-xs tracking-wider uppercase font-semibold">
          <span>06</span>
          <span>•</span>
          <span>Failure Modes & Limitations</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif text-zinc-900 pb-3 border-b border-zinc-200 mb-6">
          6. Model Limitations & Out-Of-Distribution (OOD) Failure Modes
        </h2>

        <p className="text-sm sm:text-base text-zinc-700 font-light leading-relaxed mb-6">
          In clinical machine learning, communicating what a model <strong>cannot</strong> do is as vital as publishing its accuracy metrics.
          NephroScan AI is an assistive research CADe prototype with explicit operational boundaries:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs">
            <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center mb-3">
              <AlertTriangle size={20} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-2 text-sm">Binary Closed-World Scope</h4>
            <p className="text-xs text-zinc-600 font-light leading-relaxed">
              Trained exclusively to separate <em>Calculus</em> from <em>Normal Renal Ultrasound</em>. It has never seen other abdominal organs (liver, gallbladder, spleen), and lacks open-world anomaly rejection.
            </p>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs">
            <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-700 flex items-center justify-center mb-3">
              <ShieldAlert size={20} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-2 text-sm">Laptop Screenshot Artifacts</h4>
            <p className="text-xs text-zinc-600 font-light leading-relaxed">
              Dark-mode terminals and IDEs feature bright white text against black backdrops. To convolutional filters, this high-contrast gradient pattern is mathematically identical to hyperechoic calculi, triggering false positives.
            </p>
          </div>

          <div className="bg-white p-6 rounded-3xl border border-zinc-200/90 shadow-2xs">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center mb-3">
              <ShieldCheck size={20} />
            </div>
            <h4 className="font-semibold text-zinc-900 mb-2 text-sm">Modality Guard Protection</h4>
            <p className="text-xs text-zinc-600 font-light leading-relaxed">
              General colorful photos (beaches, wallpapers, landscapes) are intercepted by acoustic chromaticity filters (mean saturation &gt; 0.22, RGB divergence &gt; 25.0) and rejected before reaching the CNN.
            </p>
          </div>
        </div>

        <div className="bg-zinc-900 text-zinc-200 rounded-3xl p-6 sm:p-8 font-light text-xs sm:text-sm leading-relaxed border border-zinc-800">
          <h4 className="font-semibold text-white mb-2 text-base flex items-center gap-2">
            <BookOpen size={18} className="text-emerald-400" />
            Key Clinical Takeaway for Evaluators
          </h4>
          <p className="mb-3 text-zinc-300">
            Never feed non-medical computer screenshots, non-renal CTs, or arbitrary camera photos to the model. Always inspect the <strong>Grad-CAM visual explanation</strong>: in an authentic scan, the activation focus must map anatomically to the renal pelvis, calyces, or parenchyma.
          </p>
          <p className="text-zinc-400 text-xs">
            NephroScan AI is not an FDA 510(k) cleared diagnostic device. Output must always be interpreted in conjunction with patient symptoms, laboratory analysis, and expert physician evaluation.
          </p>
        </div>
      </motion.section>
    </motion.div>
  );
};
