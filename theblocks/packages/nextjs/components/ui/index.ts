// Universal Animation System for PayFlow Protocol
// Re-export all animation components for easy importing

// Core Animation Components
export {
  ScrollReveal,
  PageTransition,
  SectionReveal,
  AnimatedCard,
  StaggerContainer,
  ScrollRevealGroup, // Alias for backward compatibility
  useScrollReveal,
  useParentReveal,
  ANIMATION_PRESETS,
} from "./ScrollReveal";

// Global Animation Context
export { AnimationProvider, useAnimation, Motion, Reveal } from "./AnimationProvider";

// Live Demo Widgets
export {
  PaymentFlowDemo,
  OracleConsensusDemo,
  SecurityShieldDemo,
  ComplianceDemo,
  SettlementSpeedDemo,
} from "./LiveDemoWidget";

// Narrative Storytelling System
export {
  NARRATIVE,
  StoryBlock,
  PainPointCard,
  StatCard,
  QuoteBlock,
  TimelineItem,
  MissionImpactCard,
  NarrativeDivider,
  NarrativeBadge,
} from "./Narrative";

// Modern AI-Era Effects
export {
  GlassmorphicCard,
  FloatingElement,
  MorphingBlob,
  AnimatedCounter,
  PulseRing,
  TypewriterText,
} from "./ModernEffects";

// Particle Field & Interactive Effects
export { ParticleField, FloatingOrb, MagneticButton, RippleButton } from "./ParticleField";

// Advanced Animations & Scroll Effects
export {
  ScrollProgress,
  SectionNav,
  Parallax,
  TextReveal,
  StaggeredList,
  TiltCard,
  InfiniteScroll,
  Spotlight,
} from "./AdvancedAnimations";

// Neural Network Background
export { NeuralNetworkBackground } from "./NeuralNetworkBackground";

// Magic Cursor
export { MagicCursor } from "./MagicCursor";
