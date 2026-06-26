import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import JDInputForm from './components/JDInputForm';
import CandidateCards from './components/CandidateCard';
import VoiceOnboarding from './components/VoiceOnboarding';

/* ============================================
   VaaniMatch — Main App Shell
   Tab-based navigation between:
     1. Dashboard (JD Input → Candidate Cards)
     2. Candidate Onboarding (Mobile Voice UI)
   ============================================ */

type View = 'dashboard' | 'onboarding';

// --- Navigation Tab ---
function NavTab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`relative px-6 py-2.5 text-sm font-label transition-all rounded-full ${
        active ? 'text-white' : 'text-gray-400 hover:text-gray-200'
      }`}
    >
      {active && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-0 rounded-full"
          style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.3), rgba(251,113,133,0.2))' }}
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        />
      )}
      <span className="relative z-10">{label}</span>
    </button>
  );
}

export default function App() {
  const [view, setView] = useState<View>('dashboard');
  const [showCandidates, setShowCandidates] = useState(false);

  const handleJDSubmit = (jdText: string, language: 'en' | 'hi') => {
    console.log('JD submitted:', { jdText, language });
    setShowCandidates(true);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* --- Top Navigation Bar --- */}
      <header className="glass-strong sticky top-0 z-50 px-8 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #8B5CF6, #FB7185)' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            </svg>
          </div>
          <span className="font-headline text-xl gradient-text">VaaniMatch</span>
        </div>

        <nav className="flex items-center gap-1 glass rounded-full p-1">
          <NavTab label="🎯 Recruiter Dashboard" active={view === 'dashboard'} onClick={() => { setView('dashboard'); setShowCandidates(false); }} />
          <NavTab label="🎤 Candidate Onboarding" active={view === 'onboarding'} onClick={() => setView('onboarding')} />
        </nav>

        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-label"
            style={{ background: 'linear-gradient(135deg, #22D3EE, #8B5CF6)' }}
          >
            AR
          </div>
        </div>
      </header>

      {/* --- Main Content --- */}
      <main className="flex-1 px-8 py-10">
        <AnimatePresence mode="wait">
          {view === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              {!showCandidates ? (
                <JDInputForm onSubmit={handleJDSubmit} />
              ) : (
                <div>
                  <button
                    className="mb-6 text-sm font-label text-gray-400 hover:text-violet-electric transition-colors flex items-center gap-1"
                    onClick={() => setShowCandidates(false)}
                  >
                    ← Back to JD Input
                  </button>
                  <CandidateCards />
                </div>
              )}
            </motion.div>
          )}
          {view === 'onboarding' && (
            <motion.div
              key="onboarding"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <VoiceOnboarding />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* --- Footer --- */}
      <footer className="glass-strong px-8 py-4 text-center">
        <p className="text-xs text-gray-500 font-label">
          VaaniMatch © 2026 — AI-Powered Recruitment for Bharat 🇮🇳
        </p>
      </footer>
    </div>
  );
}
