import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileType, Loader2, Activity, AlertCircle, ShieldCheck } from "lucide-react";

export default function App() {
  const [step, setStep] = useState("upload"); // 'upload' | 'processing' | 'results'

  const handleUpload = () => {
    setStep("processing");
    // Simulate AI pipeline processing time
    setTimeout(() => {
      setStep("results");
    }, 2500);
  };

  const reset = () => {
    setStep("upload");
  };

  return (
    <>
      {/* Import the Instrument Serif font for Opensource UI headers */}
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');
        .font-serif {
          font-family: 'Instrument Serif', serif;
        }
      `}} />

      <main className="min-h-screen bg-zinc-50 text-zinc-900 font-sans selection:bg-zinc-200 flex flex-col items-center justify-center p-6 sm:p-12">
        
        {/* ==================== SECTION: HERO LANDING ==================== */}
        <div className="max-w-2xl w-full text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-medium text-zinc-600 mb-6 shadow-sm">
            <ShieldCheck size={14} className="text-emerald-600" />
            HIPAA Compliant Processing
          </div>
          <h1 className="text-4xl md:text-5xl tracking-tight text-zinc-900 mb-4 font-serif">
            Renal Scan Analysis
          </h1>
          <p className="text-zinc-500 text-lg md:text-xl max-w-lg mx-auto leading-relaxed">
            Upload an ultrasound or CT cross-section for instant calculus detection and dilation grading.
          </p>
        </div>

        {/* ==================== SECTION: DIAGNOSTIC WORKSPACE ==================== */}
        <div className="max-w-2xl w-full bg-white border border-neutral-200 rounded-2xl p-2 shadow-sm min-h-[340px] relative overflow-hidden flex flex-col">
          <AnimatePresence mode="wait">
            
            {/* STATE: UPLOAD DROPZONE */}
            {step === "upload" && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
                className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-neutral-200 rounded-xl bg-zinc-50/50 hover:bg-zinc-50 hover:border-zinc-300 transition-all cursor-pointer m-2"
                onClick={handleUpload}
                whileHover={{ scale: 0.995 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="h-12 w-12 bg-white border border-neutral-200 rounded-full flex items-center justify-center mb-4 shadow-sm">
                  <UploadCloud className="text-zinc-600" size={20} />
                </div>
                <h3 className="text-base font-medium text-zinc-900 mb-1">Select or drop renal scan</h3>
                <p className="text-sm text-zinc-500">DICOM, PNG, or JPEG (Max 10MB)</p>
              </motion.div>
            )}

            {/* STATE: AI PROCESSING */}
            {step === "processing" && (
              <motion.div
                key="processing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex-1 flex flex-col items-center justify-center m-2"
              >
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  className="mb-5"
                >
                  <Loader2 className="text-zinc-400" size={36} />
                </motion.div>
                <h3 className="text-base font-medium text-zinc-900 mb-2 animate-pulse">Running localized segmentation...</h3>
                <p className="text-sm text-zinc-500">Measuring pelvic dilation and density markers.</p>
              </motion.div>
            )}

            {/* STATE: CLINICAL DASHBOARD */}
            {step === "results" && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className="flex-1 p-4 sm:p-6 flex flex-col"
              >
                {/* File Header */}
                <div className="flex items-center justify-between pb-4 mb-5 border-b border-neutral-100">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 bg-zinc-100 rounded-lg flex items-center justify-center border border-zinc-200">
                      <FileType className="text-zinc-600" size={18} />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-zinc-900">patient_scan_0942.jpeg</p>
                      <p className="text-xs text-zinc-500">Analyzed just now</p>
                    </div>
                  </div>
                  <button 
                    onClick={reset}
                    className="text-sm font-medium text-zinc-600 hover:text-zinc-900 transition-colors px-3 py-1.5 rounded-md hover:bg-zinc-100 cursor-pointer"
                  >
                    New Scan
                  </button>
                </div>

                {/* Metric Cards Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 flex-1">
                  
                  {/* Metric 1: Calculus Detection */}
                  <div className="p-5 border border-neutral-200 rounded-xl bg-zinc-50/50 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-4">
                        <Activity className="text-zinc-400" size={16} />
                        <h4 className="text-sm font-medium text-zinc-600">Calculus Detection</h4>
                      </div>
                      <div className="flex items-end gap-2 mb-2">
                        <span className="text-4xl font-semibold text-zinc-900 tracking-tight font-serif">87.4%</span>
                        <span className="text-xs text-rose-700 font-medium mb-1.5 flex items-center gap-1 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                          <AlertCircle size={12} /> Active
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-zinc-500 mt-2">High probability of calculus in lower pole.</p>
                  </div>

                  {/* Metric 2: Ureteral Dilation */}
                  <div className="p-5 border border-neutral-200 rounded-xl bg-zinc-50/50 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-4">
                        <Activity className="text-zinc-400" size={16} />
                        <h4 className="text-sm font-medium text-zinc-600">Ureteral Dilation</h4>
                      </div>
                      <div className="flex items-end gap-2 mb-2">
                        <span className="text-4xl font-semibold text-zinc-900 tracking-tight font-serif">Mild</span>
                        <span className="text-xs text-amber-700 font-medium mb-1.5 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                          Hydronephrosis
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-zinc-500 mt-2">Pelvic dilation measured at 11mm.</p>
                  </div>

                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* ==================== SECTION: USAGE & FREE TIER COUNTER ==================== */}
        <p className="mt-8 text-sm text-zinc-400">
          You have <span className="font-medium text-zinc-600">3 free scans</span> remaining today.
        </p>

      </main>
    </>
  );
}