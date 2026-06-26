import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
} from 'recharts';

/* ============================================
   Prompt 3.3 — Ranked Candidate Cards with Radial Scores
   Features:
     1. Candidate name + verified badge
     2. 5 radial progress bars (Semantic, Trajectory, Behavior, Graph, Verification)
     3. Expanding accordion "Explain Ranking"
     4. "Desired Seniority" slider with dynamic re-weighting
   ============================================ */

// --- Types ---
interface CandidateScore {
  semantic: number;
  trajectory: number;
  behavior: number;
  graph: number;
  verification: number;
}

interface Candidate {
  id: string;
  name: string;
  title: string;
  verified: boolean;
  scores: CandidateScore;
  explanation: string;
}

// --- Score color mapping ---
const SCORE_COLORS: Record<keyof CandidateScore, string> = {
  semantic: '#8B5CF6',
  trajectory: '#FB7185',
  behavior: '#22D3EE',
  graph: '#F59E0B',
  verification: '#10B981',
};

const SCORE_LABELS: Record<keyof CandidateScore, string> = {
  semantic: 'Semantic',
  trajectory: 'Trajectory',
  behavior: 'Behavior',
  graph: 'Graph',
  verification: 'Verified',
};

// --- Mini Radial Progress Component ---
function RadialScore({ value, color, label }: { value: number; color: string; label: string }) {
  const data = [{ value: value * 100, fill: color }];

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative">
        <RadialBarChart
          width={72}
          height={72}
          innerRadius="70%"
          outerRadius="100%"
          data={data}
          startAngle={90}
          endAngle={-270}
          barSize={6}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar
            dataKey="value"
            cornerRadius={4}
            background={{ fill: 'rgba(255,255,255,0.06)' }}
          />
        </RadialBarChart>
        <span className="absolute inset-0 flex items-center justify-center text-xs font-label text-white">
          {Math.round(value * 100)}
        </span>
      </div>
      <span className="text-[10px] text-gray-400 font-label uppercase tracking-wider">{label}</span>
    </div>
  );
}

// --- Verified Badge ---
function VerifiedBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-label uppercase tracking-wide"
      style={{
        background: 'rgba(34, 211, 238, 0.15)',
        color: '#22D3EE',
        border: '1px solid rgba(34, 211, 238, 0.3)',
      }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
      </svg>
      Verified
    </span>
  );
}

// --- Mock Candidates Data ---
const MOCK_CANDIDATES: Candidate[] = [
  {
    id: 'c_1',
    name: 'Priya Sharma',
    title: 'Senior ML Engineer',
    verified: true,
    scores: { semantic: 0.92, trajectory: 0.85, behavior: 0.78, graph: 0.90, verification: 1.0 },
    explanation: 'Priya\'s resume strongly aligns with the JD\'s emphasis on deep learning frameworks (PyTorch, TensorFlow). Her trajectory shows consistent upward mobility from Infosys → Google → Lead at a Series-B startup. If she had more published research papers, her Semantic score would be 0.97.',
  },
  {
    id: 'c_2',
    name: 'Rahul Mehra',
    title: 'Full Stack Developer',
    verified: false,
    scores: { semantic: 0.75, trajectory: 0.60, behavior: 0.88, graph: 0.45, verification: 0.0 },
    explanation: 'Rahul excels in behavioral engagement (active GitHub, Stack Overflow contributions). However, his career trajectory shows lateral moves without seniority increases. His Graph score is low due to limited connections to companies in the JD\'s industry.',
  },
  {
    id: 'c_3',
    name: 'Ananya Iyer',
    title: 'Data Scientist',
    verified: true,
    scores: { semantic: 0.88, trajectory: 0.95, behavior: 0.65, graph: 0.82, verification: 1.0 },
    explanation: 'Ananya\'s trajectory is exceptional — rapid progression from Junior DS to VP of Data in 6 years. Her Semantic match is high but would increase if her resume explicitly mentioned "MLOps" tooling. She is e-Shram verified with cross-validated credentials.',
  },
];

// --- Seniority weight adjustments ---
function adjustScores(scores: CandidateScore, seniority: number): number {
  // Higher seniority = more weight on trajectory and graph
  const seniorityFactor = seniority / 10;
  const weights = {
    semantic: 0.3 - seniorityFactor * 0.05,
    trajectory: 0.15 + seniorityFactor * 0.1,
    behavior: 0.2 - seniorityFactor * 0.03,
    graph: 0.15 + seniorityFactor * 0.05,
    verification: 0.2 - seniorityFactor * 0.07,
  };
  return Object.entries(scores).reduce(
    (total, [key, val]) => total + val * weights[key as keyof CandidateScore],
    0
  );
}

// --- Single Candidate Card ---
function CandidateCardItem({ candidate, seniority }: { candidate: Candidate; seniority: number }) {
  const [expanded, setExpanded] = useState(false);
  const totalScore = adjustScores(candidate.scores, seniority);

  return (
    <motion.div
      layout
      className="glass rounded-2xl p-5 card-hover"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-headline text-lg text-white">{candidate.name}</h3>
            {candidate.verified && <VerifiedBadge />}
          </div>
          <p className="text-sm text-gray-400">{candidate.title}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-headline gradient-text">{Math.round(totalScore * 100)}</div>
          <div className="text-[10px] text-gray-500 font-label uppercase">Total Score</div>
        </div>
      </div>

      {/* Radial Scores Row */}
      <div className="flex items-center justify-between mb-4 px-2">
        {(Object.keys(SCORE_COLORS) as (keyof CandidateScore)[]).map(key => (
          <RadialScore
            key={key}
            value={candidate.scores[key]}
            color={SCORE_COLORS[key]}
            label={SCORE_LABELS[key]}
          />
        ))}
      </div>

      {/* Explain Ranking Accordion */}
      <button
        className="w-full text-left text-sm font-label text-violet-electric/80 hover:text-violet-electric transition-colors flex items-center gap-2"
        onClick={() => setExpanded(!expanded)}
      >
        <motion.span
          animate={{ rotate: expanded ? 90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          ▸
        </motion.span>
        Explain Ranking
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="mt-3 p-4 rounded-xl text-sm text-gray-300 leading-relaxed"
              style={{ background: 'rgba(139, 92, 246, 0.08)', border: '1px solid rgba(139, 92, 246, 0.15)' }}
            >
              {candidate.explanation}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// --- Main Candidate Cards Dashboard ---
export default function CandidateCards() {
  const [seniority, setSeniority] = useState(5);

  // Sort candidates by adjusted total score
  const sorted = [...MOCK_CANDIDATES].sort(
    (a, b) => adjustScores(b.scores, seniority) - adjustScores(a.scores, seniority)
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="w-full max-w-3xl mx-auto"
    >
      {/* Header + Seniority Slider */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="font-headline text-3xl gradient-text">Ranked Candidates</h2>
        <div className="flex items-center gap-3 glass rounded-full px-5 py-2.5">
          <span className="text-xs font-label text-gray-400 uppercase tracking-wider">Seniority</span>
          <input
            type="range"
            min="0"
            max="10"
            value={seniority}
            onChange={e => setSeniority(Number(e.target.value))}
            className="w-24 accent-violet-electric"
          />
          <span className="text-sm font-label text-violet-electric w-6 text-center">{seniority}</span>
        </div>
      </div>

      {/* Candidate Cards */}
      <div className="flex flex-col gap-4">
        <AnimatePresence>
          {sorted.map((c, i) => (
            <motion.div
              key={c.id}
              layout
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <CandidateCardItem candidate={c} seniority={seniority} />
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
