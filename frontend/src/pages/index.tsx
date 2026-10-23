import React, { useState, useEffect } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { motion, AnimatePresence, type Variants } from "framer-motion";
import { Header } from "../components/Header";
import { EngineTab } from "../components/EngineTab";
import { DocsTab } from "../components/DocsTab";

const tabTransition: Variants = {
  initial: { opacity: 0, y: 12, filter: "blur(3px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.35, ease: "easeOut" } },
  exit: { opacity: 0, y: -12, filter: "blur(3px)", transition: { duration: 0.2, ease: "easeIn" } }
};

export default function Home() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"engine" | "docs">("engine");

  // Sync tab with query parameters (?tab=docs / ?tab=engine)
  useEffect(() => {
    if (router.isReady && router.query.tab) {
      const tabParam = String(router.query.tab).toLowerCase();
      if (tabParam === "docs") {
        setActiveTab("docs");
      } else if (tabParam === "engine") {
        setActiveTab("engine");
      }
    }
  }, [router.isReady, router.query.tab]);

  const handleTabChange = (tab: "engine" | "docs") => {
    setActiveTab(tab);
    router.push(`/?tab=${tab}`, undefined, { shallow: true });
  };

  return (
    <>
      <Head>
        <title>KidneyAI | Premium Clinical Renal Diagnostic Cockpit</title>
        <meta
          name="description"
          content="First-principles deep learning system for renal calculus detection, Grad-CAM spatial localization, and automated pelvicalyceal dilation grading."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, viewport-fit=cover" />
        <meta name="theme-color" content="#fafafa" />
      </Head>

      <div className="min-h-screen bg-[#FAFAFA] text-zinc-900 font-sans flex flex-col selection:bg-emerald-100/80">
        {/* Sticky Glass Navbar */}
        <Header activeTab={activeTab} onTabChange={handleTabChange} />

        {/* Dynamic Tab Content Area */}
        <main className="flex-1 w-full flex flex-col relative overflow-x-hidden">
          <AnimatePresence mode="wait">
            {activeTab === "engine" ? (
              <motion.div
                key="engine-screen"
                variants={tabTransition}
                initial="initial"
                animate="animate"
                exit="exit"
                className="w-full flex-1"
              >
                <EngineTab />
              </motion.div>
            ) : (
              <motion.div
                key="docs-screen"
                variants={tabTransition}
                initial="initial"
                animate="animate"
                exit="exit"
                className="w-full flex-1"
              >
                <DocsTab />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </>
  );
}
