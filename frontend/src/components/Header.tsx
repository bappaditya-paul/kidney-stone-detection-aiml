import React from "react";
import Link from "next/link";
import { ShieldCheck } from "lucide-react";

interface HeaderProps {
  activeTab: "engine" | "docs";
  onTabChange: (tab: "engine" | "docs") => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  return (
    <header className="glass-panel sticky top-0 z-50 border-b border-zinc-200/80 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => onTabChange("engine")}
            className="flex items-center gap-2 text-zinc-900 hover:opacity-90 transition-opacity text-left group"
          >
            <div className="w-9 h-9 rounded-xl bg-zinc-900 text-emerald-400 flex items-center justify-center shadow-xs border border-zinc-800 transition-transform group-hover:scale-105">
              <ShieldCheck size={20} strokeWidth={2.2} />
            </div>
            <div className="flex flex-col">
              <span className="font-serif text-xl sm:text-2xl tracking-tight leading-none text-zinc-900">
                KidneyAI
              </span>
            </div>
          </button>
          
          <div className="hidden sm:inline-flex items-center gap-1.5 bg-emerald-50 border border-emerald-200/80 px-2.5 py-1 rounded-full text-[11px] font-semibold text-emerald-800 tracking-wide">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            Clinical Engine
          </div>
        </div>

        {/* Navigation Tabs (Engine vs Docs) */}
        <nav className="flex items-center bg-zinc-100/90 p-1 rounded-full border border-zinc-200/70 shadow-2xs">
          <button
            type="button"
            onClick={() => onTabChange("engine")}
            className={`px-5 py-1.5 rounded-full text-xs sm:text-sm font-medium transition-all duration-200 touch-target-min flex items-center justify-center ${
              activeTab === "engine"
                ? "bg-white text-zinc-900 shadow-xs font-semibold"
                : "text-zinc-500 hover:text-zinc-900"
            }`}
          >
            Engine
          </button>
          <button
            type="button"
            onClick={() => onTabChange("docs")}
            className={`px-5 py-1.5 rounded-full text-xs sm:text-sm font-medium transition-all duration-200 touch-target-min flex items-center justify-center ${
              activeTab === "docs"
                ? "bg-white text-zinc-900 shadow-xs font-semibold"
                : "text-zinc-500 hover:text-zinc-900"
            }`}
          >
            Docs
          </button>
        </nav>
      </div>
    </header>
  );
};
