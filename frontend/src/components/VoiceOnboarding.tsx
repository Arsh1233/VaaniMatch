import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/* ============================================
   Prompt 3.4 — Candidate Voice Onboarding Mobile UI
   Features:
     1. Mobile-first bold gradient background
     2. Large glowing animated microphone with ripple effect
     3. Conversational chat bubble UI with progressive reveal
     4. "e-Shram Verified" badge with gold shimmer animation
   ============================================ */

// --- Chat bubble message type ---
interface ChatMessage {
  id: number;
  text: string;
  sender: 'ai' | 'user';
}

// --- Simulated conversation flow ---
const CONVERSATION_FLOW: ChatMessage[] = [
  { id: 1, text: '👋 Namaste! Welcome to VaaniMatch. Tap the mic and tell me about yourself.', sender: 'ai' },
  { id: 2, text: 'मेरा नाम राहुल है, मैं Software Engineer हूँ...', sender: 'user' },
  { id: 3, text: 'Got it! You are a Software Engineer. How many years of experience?', sender: 'ai' },
  { id: 4, text: '5 साल का experience है, TCS में काम किया है।', sender: 'user' },
  { id: 5, text: '✅ Noted — 5 years at TCS. What are your top skills?', sender: 'ai' },
  { id: 6, text: 'Python, Machine Learning, और SQL', sender: 'user' },
  { id: 7, text: '🎯 Perfect! Python, Machine Learning, SQL added. Let\'s verify your identity via e-Shram.', sender: 'ai' },
];

// --- Ripple Animation Component ---
function Ripple({ isActive }: { isActive: boolean }) {
  if (!isActive) return null;
  return (
    <>
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-full border-2 border-violet-electric/40"
          initial={{ scale: 1, opacity: 0.6 }}
          animate={{ scale: 2.5, opacity: 0 }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: i * 0.4,
            ease: 'easeOut',
          }}
        />
      ))}
    </>
  );
}

// --- Chat Bubble ---
function ChatBubble({ message }: { message: ChatMessage }) {
  const isAI = message.sender === 'ai';
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isAI ? 'justify-start' : 'justify-end'}`}
    >
      <div
        className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
          isAI
            ? 'glass rounded-tl-sm text-gray-200'
            : 'rounded-tr-sm text-white'
        }`}
        style={
          !isAI
            ? { background: 'linear-gradient(135deg, #8B5CF6, #FB7185)' }
            : undefined
        }
      >
        {message.text}
      </div>
    </motion.div>
  );
}

// --- e-Shram Verified Badge ---
function EShramBadge() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="flex items-center justify-center gap-2 px-6 py-3 rounded-full glass mx-auto"
      style={{ border: '1px solid rgba(245, 158, 11, 0.4)' }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="#F59E0B">
        <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
      </svg>
      <span className="badge-gold font-headline text-lg">e-Shram Verified</span>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="#10B981">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
      </svg>
    </motion.div>
  );
}

// --- Main Voice Onboarding Component ---
export default function VoiceOnboarding() {
  const [isListening, setIsListening] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([CONVERSATION_FLOW[0]]);
  const [messageIndex, setMessageIndex] = useState(1);
  const [aadhaarLinked, setAadhaarLinked] = useState(false);
  const [otpInput, setOtpInput] = useState('');
  const [showOtpField, setShowOtpField] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Simulate voice conversation
  const handleMicTap = () => {
    if (aadhaarLinked) return;
    setIsListening(true);

    setTimeout(() => {
      setIsListening(false);
      if (messageIndex < CONVERSATION_FLOW.length) {
        // Add next user message
        setMessages(prev => [...prev, CONVERSATION_FLOW[messageIndex]]);
        setMessageIndex(prev => prev + 1);

        // Add AI response after delay
        setTimeout(() => {
          if (messageIndex + 1 < CONVERSATION_FLOW.length) {
            setMessages(prev => [...prev, CONVERSATION_FLOW[messageIndex + 1]]);
            setMessageIndex(prev => prev + 1);
          }
          // After last message, show OTP field
          if (messageIndex + 1 >= CONVERSATION_FLOW.length - 1) {
            setTimeout(() => setShowOtpField(true), 800);
          }
        }, 800);
      }
    }, 1200);
  };

  const handleOtpVerify = () => {
    if (otpInput.length === 6) {
      setShowOtpField(false);
      setAadhaarLinked(true);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto flex flex-col items-center min-h-screen py-8 px-4">
      {/* --- Header --- */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="font-headline text-4xl gradient-text mb-2">VaaniMatch</h1>
        <p className="text-sm text-gray-400">Voice-First Candidate Registration</p>
      </motion.div>

      {/* --- Glowing Microphone Button --- */}
      <motion.div className="relative mb-8">
        <Ripple isActive={isListening} />
        <motion.button
          className="relative z-10 w-28 h-28 rounded-full flex items-center justify-center text-white"
          style={{
            background: isListening
              ? 'linear-gradient(135deg, #FB7185, #8B5CF6)'
              : 'linear-gradient(135deg, #8B5CF6, #22D3EE)',
            animation: !isListening ? 'micPulse 2s ease-in-out infinite' : 'none',
          }}
          whileTap={{ scale: 0.9 }}
          whileHover={{ scale: 1.05 }}
          onClick={handleMicTap}
          disabled={aadhaarLinked}
          aria-label="Tap to speak"
        >
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
          </svg>
        </motion.button>
        <p className="text-center text-xs text-gray-500 mt-3 font-label">
          {isListening ? '🎙️ Listening...' : aadhaarLinked ? '✅ Profile Complete' : 'Tap to speak'}
        </p>
      </motion.div>

      {/* --- Chat Bubbles --- */}
      <div
        ref={chatRef}
        className="w-full glass rounded-2xl p-4 flex flex-col gap-3 mb-6 overflow-y-auto"
        style={{ maxHeight: '320px', minHeight: '200px' }}
      >
        <AnimatePresence>
          {messages.map(msg => (
            <ChatBubble key={msg.id} message={msg} />
          ))}
        </AnimatePresence>
      </div>

      {/* --- OTP Verification Field --- */}
      <AnimatePresence>
        {showOtpField && !aadhaarLinked && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            className="w-full glass rounded-2xl p-5 mb-6"
          >
            <p className="text-sm text-gray-300 mb-3 font-label">🔐 Enter Aadhaar OTP to verify via e-Shram</p>
            <div className="flex gap-2">
              <input
                type="text"
                maxLength={6}
                placeholder="● ● ● ● ● ●"
                value={otpInput}
                onChange={e => setOtpInput(e.target.value.replace(/\D/g, ''))}
                className="flex-1 glass rounded-xl px-4 py-3 text-center text-lg font-label tracking-[0.5em] neon-focus bg-transparent text-white"
              />
              <button
                className="btn-neon px-5"
                onClick={handleOtpVerify}
                disabled={otpInput.length !== 6}
              >
                Verify
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* --- e-Shram Verified Badge --- */}
      <AnimatePresence>
        {aadhaarLinked && <EShramBadge />}
      </AnimatePresence>
    </div>
  );
}
