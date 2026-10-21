import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  UploadCloud, FileType, Loader2, Activity, AlertCircle, 
  Download, Github, Database, BookOpen, ShieldCheck 
} from "lucide-react";

export default function App() {
  const [step, setStep] = useState("upload"); // 'upload' | 'processing' | 'results'
  
  // Mocking sound effects as requested
  const playSound = (type) => {
    console.log(`[Audio Mock] Playing sound: ${type}`);
  };

  const handleUpload = () => {
    playSound("soft-click.mp3");
    setStep("processing");
    
    // Simulate AI pipeline delay (2.5 seconds)
    setTimeout(() => {
      setStep("results");
      playSound("chime-resolve.mp3");
    }, 2500);
  };

  const handleDownload = () => {
    playSound("soft-click.mp3");
    alert("Downloading Clinical Report...");
  };

  const reset = () => {
    playSound("soft-click.mp3");
    setStep("upload");
  };

  // Cleanup timeout just in case (best practice)
  useEffect(() => {
    return () => {
      // Clear any pending timeouts on unmount if we added a ref, 
      // but simple enough here for the prototype.
    };
  }, []);

  return (
    <>
      {/* Injecting Opensource UI standard typography (Instrument Serif for headers) */}
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');
        .font-serif {
          font-family: 'Instrument Serif', serif;
        }
      `}} />

      <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans selection:bg-zinc-200 flex flex-col">
        
        {/* ==================== SECTION: TOP NAVIGATION ==================== */}
        <header className="w-full border-b border-neutral-200 bg-white/80 backdrop-blur-md sticky top-0 z-10">
          <div className="max-w-5xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2 cursor-pointer" onClick={reset}>
              <ShieldCheck className="text-emerald-600" size={24} />
              <span className="font-serif text-2xl tracking-tight mt-1">
                KidneyAI
              </span>
            </div>
            
            <nav className="hidden sm:flex items-center gap-6 text-sm font-medium text-zinc-500">
              <a href="#docs" className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors">
                <BookOpen size={16} /> Docs
              </a>
              <a href="#dataset" className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors">
                <Database size={16} /> Dataset
              </a>
              <a href="#github" className="flex items-center gap-1.5 hover:text-zinc-900 transition-colors">
                <Github size={16} /> GitHub
              </a>
            </nav>
          </div>
        </header>

        {/* ==================== SECTION: MAIN WORKSPACE ==================== */}
        <main className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12">
          
          <div className="max-w-2xl w-full text-center mb-8">
            <h1 className="text-4xl md:text-5xl tracking-tight text-zinc-900 mb-3 font-serif">
              Renal Scan Analysis
            </h1>
            <p className="text-zinc-500 text-lg">
              Upload an ultrasound or CT cross-section for instant calculus detection and grading.
            </p>
          </div>

          <div className="max-w-2xl w-full bg-white border border-neutral-200 rounded-2xl p-2 shadow-sm min-h-[380px] relative overflow-hidden flex flex-col">
            <AnimatePresence mode="wait">
              
              {/* STATE 1: UPLOAD */}
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
                  <div className="h-14 w-14 bg-white border border-neutral-200 rounded-full flex items-center justify-center mb-5 shadow-sm">
                    <UploadCloud className="text-zinc-600" size={24} />
                  </div>
                  <h3 className="text-lg font-medium text-zinc-900 mb-1">Select or drop renal scan</h3>
                  <p className="text-sm text-zinc-500">DICOM, PNG, or JPEG (Max 10MB)</p>
                </motion.div>
              )}

              {/* STATE 2: PROCESSING */}
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
                    transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                    className="mb-6"
                  >
                    <Loader2 className="text-zinc-400" size={40} />
                  </motion.div>
                  <h3 className="text-lg font-medium text-zinc-900 mb-2 animate-pulse">Running localized segmentation...</h3>
                  <p className="text-sm text-zinc-500">Measuring pelvic dilation and density markers.</p>
                </motion.div>
              )}

              {/* STATE 3: RESULTS */}
              {step === "results" && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  className="flex-1 p-4 sm:p-6 flex flex-col justify-between"
                >
                  <div>
                    {/* File Header */}
                    <div className="flex items-center justify-between pb-5 mb-5 border-b border-neutral-100">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 bg-zinc-100 rounded-lg flex items-center justify-center border border-zinc-200">
                          <FileType className="text-zinc-600" size={18} />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-zinc-900">patient_scan_0942.jpeg</p>
                          <p className="text-xs text-zinc-500">Analyzed just now</p>
                        </div>
                      </div>
                    </div>

                    {/* Metric Cards Grid */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      
                        {/* Metric 1 */}
                        <div className="p-5 border border-neutral-200 rounded-xl bg-zinc-50/50">
                          <div className="flex items-center gap-2 mb-4">
                            <Activity className="text-zinc-400" size={16} />
                            <h4 className="text-sm font-medium text-zinc-600">Calculus Detection</h4>
                          </div>
                          <div className="flex items-end gap-2 mb-2">
                            <span className="text-4xl font-semibold text-zinc-900 tracking-tight font-serif">87.4%</span>
                            <span className="text-xs text-rose-700 font-medium mb-1.5 flex items-center gap-1 bg-rose-50 px-2 py-0.5 rounded-md border border-rose-200">
                              <AlertCircle size={12} /> Active
                            </span>
                          </div>
                          <p className="text-sm text-zinc-500 mt-1">High probability of calculus in lower pole.</p>
                        </div>

                        {/* Metric 2 */}
                        <div className="p-5 border border-neutral-200 rounded-xl bg-zinc-50/50">
                          <div className="flex items-center gap-2 mb-4">
                            <Activity className="text-zinc-400" size={16} />
                            <h4 className="text-sm font-medium text-zinc-600">Ureteral Dilation</h4>
                          </div>
                          <div className="flex items-end gap-2 mb-2">
                            <span className="text-4xl font-semibold text-zinc-900 tracking-tight font-serif">Mild</span>
                            <span className="text-xs text-amber-700 font-medium mb-1.5 bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
                              Hydronephrosis
                            </span>
                          </div>
                          <p className="text-sm text-zinc-500 mt-1">Pelvic dilation measured at 11mm.</p>
                        </div>
                    </div>
                  </div>

                  {/* ACTION BUTTONS (Download Report & Reset) */}
                  <div className="mt-6 pt-5 flex items-center justify-end gap-3 border-t border-neutral-100">
                    <button 
                      onClick={reset}
                      className="px-4 py-2 text-sm font-medium text-zinc-600 hover:text-zinc-900 bg-white border border-neutral-200 rounded-md shadow-sm hover:bg-zinc-50 transition-colors"
                    >
                      Analyze New Scan
                    </button>
                    <button 
                      onClick={handleDownload}
                      className="px-4 py-2 text-sm font-medium text-white bg-zinc-900 hover:bg-zinc-800 rounded-md shadow-sm flex items-center gap-2 transition-colors"
                    >
                      <Download size={16} />
                      Download Report
                    </button>
                  </div>

                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>
      </div>
    </>
  );
}