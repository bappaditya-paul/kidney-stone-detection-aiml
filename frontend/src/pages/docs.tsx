import React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { Header } from "@/components/Header";
import { DocsTab } from "@/components/DocsTab";

export default function DocsPage() {
  const router = useRouter();

  const handleTabChange = (tab: "engine" | "docs") => {
    if (tab === "engine") {
      router.push("/?tab=engine");
    }
  };

  return (
    <>
      <Head>
        <title>NephroScan AI Specs & Architecture Guide | KidneyAI Docs</title>
        <meta
          name="description"
          content="Comprehensive first-principles engineering and clinical documentation for KidneyAI: Modality Guard, MobileNetV2 architecture, Grad-CAM XAI, and FastAPI microservice."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, viewport-fit=cover" />
        <meta name="theme-color" content="#fafafa" />
      </Head>

      <div className="min-h-screen bg-[#FAFAFA] text-zinc-900 font-sans flex flex-col selection:bg-emerald-100/80">
        <Header activeTab="docs" onTabChange={handleTabChange} />

        <main className="flex-1 w-full flex flex-col relative overflow-x-hidden">
          <DocsTab />
        </main>
      </div>
    </>
  );
}
